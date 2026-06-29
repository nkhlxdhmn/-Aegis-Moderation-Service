from __future__ import annotations

from unittest.mock import patch


def test_surya_text_is_returned() -> None:
    with patch("backend.pipeline.ocr.run_surya_ocr", return_value=["primary text"]):
        from backend.pipeline.ocr import extract_ocr_text

        result = extract_ocr_text("image.jpg")

    assert result == "primary text"


def test_surya_empty_output_returns_empty_string() -> None:
    with patch("backend.pipeline.ocr.run_surya_ocr", return_value=[]):
        from backend.pipeline.ocr import extract_ocr_text

        result = extract_ocr_text("image.jpg")

    assert result == ""


def test_surya_exception_returns_empty_string() -> None:
    with patch("backend.pipeline.ocr.run_surya_ocr", side_effect=RuntimeError("surya boom")):
        from backend.pipeline.ocr import extract_ocr_text

        result = extract_ocr_text("image.jpg")

    assert result == ""
