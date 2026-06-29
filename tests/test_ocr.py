"""Tests for the Surya-only OCR pipeline."""

from __future__ import annotations

import io
import os
import tempfile
from unittest import TestCase
from unittest.mock import patch


def _write_blank_png() -> str:
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), color=(255, 255, 255)).save(buf, format="PNG")
    fd, path = tempfile.mkstemp(suffix=".png")
    os.write(fd, buf.getvalue())
    os.close(fd)
    return path


class TestSuryaOnlyOCR(TestCase):
    def test_surya_result_returned(self) -> None:
        tmp_path = _write_blank_png()
        try:
            with patch(
                "backend.pipeline.ocr.run_surya_ocr",
                return_value=["Hello World", "Test text from Surya"],
            ) as mock_surya:
                from backend.pipeline.ocr import extract_ocr_text

                result = extract_ocr_text(tmp_path)

            mock_surya.assert_called_once_with(tmp_path)
            self.assertIn("Hello World", result)
            self.assertIn("Test text from Surya", result)
        finally:
            os.unlink(tmp_path)

    def test_surya_result_deduplicates_fragments(self) -> None:
        tmp_path = _write_blank_png()
        try:
            with patch("backend.pipeline.ocr.run_surya_ocr", return_value=["hello", "Hello"]):
                from backend.pipeline.ocr import extract_ocr_text

                result = extract_ocr_text(tmp_path)

            self.assertEqual(result.lower().count("hello"), 1)
        finally:
            os.unlink(tmp_path)

    def test_surya_empty_returns_empty_string(self) -> None:
        with patch("backend.pipeline.ocr.run_surya_ocr", return_value=[]):
            from backend.pipeline.ocr import extract_ocr_text

            result = extract_ocr_text("/nonexistent/totally/missing.jpg")

        self.assertEqual(result, "")

    def test_surya_exception_returns_empty_string(self) -> None:
        with patch("backend.pipeline.ocr.run_surya_ocr", side_effect=RuntimeError("boom")):
            from backend.pipeline.ocr import extract_ocr_text

            result = extract_ocr_text("image.jpg")

        self.assertEqual(result, "")

    def test_extract_always_returns_string(self) -> None:
        with patch("backend.pipeline.ocr.run_surya_ocr", return_value=[]):
            from backend.pipeline.ocr import extract_ocr_text

            result = extract_ocr_text("image.jpg")

        self.assertIsInstance(result, str)


class TestSuryaEngine(TestCase):
    def test_run_surya_ocr_returns_empty_when_no_predictor(self) -> None:
        import backend.pipeline.surya_ocr as _mod

        original = _mod._predictor
        try:
            _mod._predictor = None
            with patch.object(_mod, "load_surya", return_value=False):
                result = _mod.run_surya_ocr("/some/image.jpg")
            self.assertEqual(result, [])
        finally:
            _mod._predictor = original

    def test_is_available_reflects_predictor_state(self) -> None:
        import backend.pipeline.surya_ocr as _mod

        original = _mod._predictor
        try:
            _mod._predictor = None
            self.assertFalse(_mod.is_available())
            _mod._predictor = object()
            self.assertTrue(_mod.is_available())
        finally:
            _mod._predictor = original

    def test_load_surya_returns_bool(self) -> None:
        import backend.pipeline.surya_ocr as _mod

        original = _mod._predictor
        try:
            _mod._predictor = None
            result = _mod.load_surya()
            self.assertIsInstance(result, bool)
        finally:
            _mod._predictor = original


class TextQualityTests(TestCase):
    def test_clean_text_scores_zero(self) -> None:
        from backend.pipeline.ocr import get_text_quality_score

        score = get_text_quality_score(
            "Brihadeeswarar Temple inscription",
            "Ancient Chola heritage",
        )
        self.assertEqual(score, 0.0)

    def test_spam_and_promotional_text_scores_higher(self) -> None:
        from backend.pipeline.ocr import get_text_quality_score

        score = get_text_quality_score(
            "Click here subscribe now",
            "Buy now sale http://a.test http://b.test http://c.test",
        )
        self.assertGreaterEqual(score, 0.6)

    def test_fake_history_text_scores_higher(self) -> None:
        from backend.pipeline.ocr import get_text_quality_score

        score = get_text_quality_score(
            "Aliens built this temple proof historians lied",
            None,
        )
        self.assertGreaterEqual(score, 0.35)

    def test_empty_inputs_score_zero(self) -> None:
        from backend.pipeline.ocr import get_text_quality_score

        self.assertEqual(get_text_quality_score("", None), 0.0)
        self.assertEqual(get_text_quality_score("", ""), 0.0)

    def test_score_is_clamped_to_unit_interval(self) -> None:
        from backend.pipeline.ocr import get_text_quality_score

        score = get_text_quality_score(
            "free money guaranteed income click here subscribe now buy now "
            "sale discount http://a.test http://b.test http://c.test "
            "aliens built this temple",
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
