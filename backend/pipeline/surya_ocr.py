"""Surya OCR engine — primary OCR for the moderation pipeline.

Loads the surya-ocr pip package (not a Docker sidecar, not the surya-ocr-main
folder). GPU/CPU selection is automatic. Returns plain-text fragments to the
ocr.py router; nothing surya-internal leaks beyond this file.

If the package is absent or initialization fails, load_surya() returns False
and callers return empty OCR text without switching engines.

Preprocessing (Phase 3):
  preprocess_for_ocr() applies a lightweight pipeline before Surya inference:
    1. Auto-orient: fix EXIF rotation
    2. Grayscale → CLAHE contrast enhancement
    3. Mild Gaussian denoise
    4. Back to RGB for Surya (which expects 3-channel input)
  The step is skipped gracefully when opencv is absent.
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


def preprocess_for_ocr(img: Any) -> Any:
    """Apply lightweight image preprocessing to improve OCR accuracy.

    Steps (each skipped on error so the original image is always returned):
      - EXIF auto-orientation
      - Grayscale conversion + CLAHE contrast enhancement
      - Gaussian denoising
      - Adaptive thresholding for very low-contrast images
      - Return as RGB PIL Image

    Falls back to the original image if opencv is unavailable or any step fails.
    """
    try:
        import cv2
        import numpy as np
        from PIL import ImageOps

        # Auto-orient using EXIF data
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # CLAHE contrast enhancement — improves low-contrast text
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Mild Gaussian denoise (small kernel to avoid blurring text strokes)
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Convert back to RGB PIL Image (Surya expects RGB)
        rgb = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        from PIL import Image as PILImage

        return PILImage.fromarray(rgb)

    except Exception:
        logger.debug("OCR preprocessing skipped (opencv unavailable or error)", exc_info=True)
        return img


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
    OCR router can fail closed without switching engines.

    Phase 3: applies preprocess_for_ocr() before inference to improve accuracy
    on low-contrast, blurry, or poorly-oriented images.
    """
    predictor = _get_predictor()
    if predictor is None:
        return []

    try:
        from PIL import Image as PILImage

        img = PILImage.open(image_path).convert("RGB")
        img = preprocess_for_ocr(img)
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
