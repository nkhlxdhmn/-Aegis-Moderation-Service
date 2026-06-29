"""Tests for the hybrid OCR pipeline (Surya primary + EasyOCR fallback).

Test cases:
  1. Surya available    â€” Surya runs, EasyOCR is NOT called.
  2. Surya unavailable  â€” EasyOCR fallback is called and returns text.
  3. Empty image        â€” returns "" without crashing.
  4. Bad image path     â€” returns "" without crashing.

Patching note: pipeline/ocr.py binds names at import time via
  from backend.pipeline.surya_ocr import run_surya_ocr
  from backend.pipeline.easyocr_engine import run_easyocr
so patches must target pipeline.ocr.run_surya_ocr / pipeline.ocr.run_easyocr,
NOT the source-module paths.
"""

from __future__ import annotations

import io
import os
import tempfile
from unittest import TestCase
from unittest.mock import patch


# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _write_blank_png() -> str:
    """Write a 64Ã—64 white PNG to a temp file and return the path."""
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.new("RGB", (64, 64), color=(255, 255, 255)).save(buf, format="PNG")
    fd, path = tempfile.mkstemp(suffix=".png")
    os.write(fd, buf.getvalue())
    os.close(fd)
    return path


# â”€â”€ Test 1: Surya available â†’ EasyOCR NOT called â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestSuryaAvailablePath(TestCase):
    def test_surya_result_returned_easyocr_not_called(self) -> None:
        """When Surya returns fragments, EasyOCR must not be called at all."""
        tmp_path = _write_blank_png()
        surya_fragments = ["Hello World", "Test text from Surya"]

        try:
            with patch("backend.pipeline.ocr.run_surya_ocr", return_value=surya_fragments) as mock_surya, \
                 patch("backend.pipeline.ocr.run_easyocr") as mock_easyocr:

                from backend.pipeline.ocr import extract_ocr_text
                result = extract_ocr_text(tmp_path)

            mock_surya.assert_called_once_with(tmp_path)
            mock_easyocr.assert_not_called()
            self.assertIn("Hello World", result)
            self.assertIn("Test text from Surya", result)
        finally:
            os.unlink(tmp_path)

    def test_surya_result_deduplicates_fragments(self) -> None:
        """Duplicate fragments (case-insensitive) must appear only once."""
        tmp_path = _write_blank_png()

        try:
            with patch("backend.pipeline.ocr.run_surya_ocr", return_value=["hello", "Hello", "HELLO"]), \
                 patch("backend.pipeline.ocr.run_easyocr") as mock_easyocr:

                from backend.pipeline.ocr import extract_ocr_text
                result = extract_ocr_text(tmp_path)

            mock_easyocr.assert_not_called()
            self.assertEqual(result.lower().count("hello"), 1)
        finally:
            os.unlink(tmp_path)


# â”€â”€ Test 2: Surya unavailable â†’ EasyOCR fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestSuryaUnavailableFallback(TestCase):
    def test_easyocr_runs_when_surya_returns_empty(self) -> None:
        """When Surya returns [], EasyOCR fallback must be called."""
        tmp_path = _write_blank_png()
        easyocr_fragments = ["Indic text from EasyOCR"]

        try:
            with patch("backend.pipeline.ocr.run_surya_ocr", return_value=[]) as mock_surya, \
                 patch("backend.pipeline.ocr.run_easyocr", return_value=easyocr_fragments) as mock_easyocr:

                from backend.pipeline.ocr import extract_ocr_text
                result = extract_ocr_text(tmp_path)

            mock_surya.assert_called_once_with(tmp_path)
            mock_easyocr.assert_called_once_with(tmp_path)
            self.assertIn("Indic text from EasyOCR", result)
        finally:
            os.unlink(tmp_path)

    def test_easyocr_runs_when_surya_returns_no_text(self) -> None:
        """When Surya returns [] (unavailable/failed), EasyOCR is the fallback."""
        tmp_path = _write_blank_png()

        try:
            with patch("backend.pipeline.ocr.run_surya_ocr", return_value=[]), \
                 patch("backend.pipeline.ocr.run_easyocr", return_value=["fallback text"]) as mock_easyocr:

                from backend.pipeline.ocr import extract_ocr_text
                result = extract_ocr_text(tmp_path)

            mock_easyocr.assert_called_once()
            self.assertIn("fallback text", result)
        finally:
            os.unlink(tmp_path)

    def test_returns_empty_string_when_both_engines_produce_nothing(self) -> None:
        """When both Surya and EasyOCR return [], result must be ''."""
        tmp_path = _write_blank_png()

        try:
            with patch("backend.pipeline.ocr.run_surya_ocr", return_value=[]), \
                 patch("backend.pipeline.ocr.run_easyocr", return_value=[]):

                from backend.pipeline.ocr import extract_ocr_text
                result = extract_ocr_text(tmp_path)

            self.assertEqual(result, "")
        finally:
            os.unlink(tmp_path)


# â”€â”€ Test 3: Empty image (blank PNG) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestEmptyImage(TestCase):
    def test_blank_image_does_not_crash(self) -> None:
        """A valid but blank image must return '' without raising."""
        tmp_path = _write_blank_png()

        try:
            with patch("backend.pipeline.ocr.run_surya_ocr", return_value=[]), \
                 patch("backend.pipeline.ocr.run_easyocr", return_value=[]):

                from backend.pipeline.ocr import extract_ocr_text
                result = extract_ocr_text(tmp_path)

            self.assertIsInstance(result, str)
            self.assertEqual(result, "")
        finally:
            os.unlink(tmp_path)

    def test_blank_image_returns_string_type(self) -> None:
        """extract_ocr_text always returns str, never None."""
        tmp_path = _write_blank_png()

        try:
            with patch("backend.pipeline.ocr.run_surya_ocr", return_value=[]), \
                 patch("backend.pipeline.ocr.run_easyocr", return_value=[]):

                from backend.pipeline.ocr import extract_ocr_text
                result = extract_ocr_text(tmp_path)

            self.assertIsInstance(result, str)
        finally:
            os.unlink(tmp_path)


# â”€â”€ Test 4: Bad image path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestBadImagePath(TestCase):
    def test_nonexistent_path_returns_empty_string(self) -> None:
        """A path that does not exist must return '' without raising."""
        with patch("backend.pipeline.ocr.run_surya_ocr", return_value=[]), \
             patch("backend.pipeline.ocr.run_easyocr", return_value=[]):

            from backend.pipeline.ocr import extract_ocr_text
            result = extract_ocr_text("/nonexistent/totally/missing.jpg")

        self.assertIsInstance(result, str)
        self.assertEqual(result, "")

    def test_invalid_file_content_returns_empty_string(self) -> None:
        """A file with garbage content must return '' without raising."""
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.write(fd, b"not an image at all")
        os.close(fd)

        try:
            with patch("backend.pipeline.ocr.run_surya_ocr", return_value=[]), \
                 patch("backend.pipeline.ocr.run_easyocr", return_value=[]):

                from backend.pipeline.ocr import extract_ocr_text
                result = extract_ocr_text(path)

            self.assertIsInstance(result, str)
            self.assertEqual(result, "")
        finally:
            os.unlink(path)

    def test_surya_engine_handles_nonexistent_path(self) -> None:
        """run_surya_ocr() must return [] for a nonexistent path, never raise."""
        from backend.pipeline.surya_ocr import run_surya_ocr
        result = run_surya_ocr("/nonexistent/totally/missing.jpg")
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    def test_easyocr_engine_handles_nonexistent_path(self) -> None:
        """run_easyocr() must return [] for a nonexistent path, never raise."""
        from backend.pipeline.easyocr_engine import run_easyocr
        result = run_easyocr("/nonexistent/totally/missing.jpg")
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])


# â”€â”€ Surya engine unit tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestSuryaEngine(TestCase):
    def test_run_surya_ocr_returns_empty_when_no_predictor(self) -> None:
        """run_surya_ocr() returns [] when predictor is None (package absent)."""
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
        """is_available() returns True iff _predictor is not None."""
        import backend.pipeline.surya_ocr as _mod
        original = _mod._predictor

        try:
            _mod._predictor = None
            self.assertFalse(_mod.is_available())

            _mod._predictor = object()  # stand-in for a real predictor
            self.assertTrue(_mod.is_available())
        finally:
            _mod._predictor = original

    def test_load_surya_returns_bool(self) -> None:
        """load_surya() must return a bool in all cases."""
        import backend.pipeline.surya_ocr as _mod
        original = _mod._predictor

        try:
            _mod._predictor = None
            result = _mod.load_surya()
            self.assertIsInstance(result, bool)
        finally:
            _mod._predictor = original


# â”€â”€ EasyOCR engine unit tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestEasyOCREngine(TestCase):
    def test_run_easyocr_returns_empty_when_no_readers(self) -> None:
        """run_easyocr() returns [] when no readers are loaded."""
        import backend.pipeline.easyocr_engine as _mod
        original = _mod._readers[:]

        try:
            _mod._readers.clear()
            with patch.object(_mod, "load_easyocr", return_value=False):
                result = _mod.run_easyocr("/some/image.jpg")
            self.assertEqual(result, [])
        finally:
            _mod._readers.clear()
            _mod._readers.extend(original)

    def test_is_available_false_when_readers_empty(self) -> None:
        """is_available() returns False when reader list is empty."""
        import backend.pipeline.easyocr_engine as _mod
        original = _mod._readers[:]

        try:
            _mod._readers.clear()
            self.assertFalse(_mod.is_available())
        finally:
            _mod._readers.clear()
            _mod._readers.extend(original)

    def test_is_available_true_when_readers_present(self) -> None:
        """is_available() returns True when reader list is non-empty."""
        import backend.pipeline.easyocr_engine as _mod
        original = _mod._readers[:]

        try:
            _mod._readers.clear()
            _mod._readers.append(object())  # stand-in for a real reader
            self.assertTrue(_mod.is_available())
        finally:
            _mod._readers.clear()
            _mod._readers.extend(original)


# â”€â”€ Text quality scoring tests (unchanged behaviour) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
