"""Surya OCR engine — primary OCR for the moderation pipeline.

Loads the surya-ocr pip package (not a Docker sidecar, not the surya-ocr-main
folder). GPU/CPU selection is automatic. Returns plain-text fragments to the
ocr.py router; nothing surya-internal leaks beyond this file.

If the package is absent or initialization fails, load_surya() returns False
and callers return empty OCR text without switching engines.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MIN_CONFIDENCE = 0.25

_predictor: Any = None
_manager: Any = None
_lock = threading.Lock()


def load_surya() -> bool:
    """Initialize the Surya predictor. Returns True on success, False otherwise.

    Thread-safe; safe to call multiple times — loads only once.
    Logs whether Surya OCR loaded or is unavailable.
    """
    global _predictor, _manager
    if _predictor is not None:
        return True

    with _lock:
        if _predictor is not None:
            return True

        try:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor

            _manager = SuryaInferenceManager()
            _predictor = RecognitionPredictor(_manager)
            logger.info("Surya OCR loaded (manager=%s)", type(_manager).__name__)
            return True

        except ImportError:
            logger.info("surya package not installed - Surya OCR unavailable")
            return False

        except Exception:
            logger.exception("Surya OCR unavailable")
            return False


def _get_predictor() -> Any:
    if _predictor is not None:
        return _predictor
    load_surya()
    return _predictor


def run_surya_ocr(image_path: str) -> list[str]:
    """Run Surya OCR on an image file. Returns text fragments or [] on any failure.

    Does not raise — all errors are logged and an empty list is returned so the
    the OCR router can fail closed without switching engines.
    """
    predictor = _get_predictor()
    if predictor is None:
        return []

    try:
        from PIL import Image as PILImage

        img = PILImage.open(image_path).convert("RGB")
        results = predictor([img])

        fragments: list[str] = []
        if results:
            blocks = getattr(results[0], "blocks", None) or []
            for block in blocks:
                if getattr(block, "skipped", False) or getattr(block, "error", False):
                    continue
                if getattr(block, "confidence", 1.0) < _MIN_CONFIDENCE:
                    continue
                text = re.sub(r"<[^>]+>", " ", getattr(block, "html", "") or "").strip()
                if text:
                    fragments.append(text)

        logger.info(
            "Surya OCR extracted %d blocks from %s",
            len(fragments),
            Path(image_path).name,
        )
        return fragments

    except Exception:
        logger.exception("Surya OCR inference failed for %s", Path(image_path).name)
        return []


def is_available() -> bool:
    """Return True if the Surya predictor is loaded and ready."""
    return _predictor is not None
