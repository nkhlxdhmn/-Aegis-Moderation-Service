"""EasyOCR engine — Indic-script fallback OCR for the moderation pipeline.

Owns all EasyOCR reader singletons and the image-preprocessing variants that
improve accuracy on memes, posters, and compressed social-media images.

Used by pipeline/ocr.py when Surya OCR produces no text, and by model_warmup.py
to pre-load readers at container startup.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# One reader per script family — EasyOCR cannot mix Indic scripts in one reader.
_LANGUAGE_GROUPS: list[list[str]] = [
    ["en", "hi"],  # Devanagari — Hindi, Marathi, Sanskrit (primary; runs all variants)
    ["te", "en"],  # Telugu
    ["kn", "en"],  # Kannada
    ["bn", "en"],  # Bengali
    ["ur", "en"],  # Urdu
]

_readers: list[Any] = []
_lock = threading.Lock()


def load_easyocr() -> bool:
    """Initialize all EasyOCR readers. Returns True if at least one reader loaded.

    Thread-safe; safe to call multiple times — loads only once.
    """
    global _readers
    if _readers:
        return True

    with _lock:
        if _readers:
            return True

        try:
            import easyocr
            import torch

            use_gpu = torch.cuda.is_available()
            logger.info(
                "Initialising %d EasyOCR readers (gpu=%s)",
                len(_LANGUAGE_GROUPS),
                use_gpu,
            )

            built: list[Any] = []
            for langs in _LANGUAGE_GROUPS:
                try:
                    reader = easyocr.Reader(
                        langs,
                        gpu=use_gpu,
                        download_enabled=True,
                        verbose=False,
                    )
                    built.append(reader)
                    logger.info("EasyOCR reader ready: %s", langs)
                except Exception:
                    logger.exception("EasyOCR reader failed for langs=%s — skipping", langs)

            _readers = built
            logger.info("EasyOCR: %d/%d readers loaded", len(built), len(_LANGUAGE_GROUPS))
            return bool(built)

        except Exception:
            logger.exception("EasyOCR initialization failed")
            return False


def _get_readers() -> list[Any]:
    if _readers:
        return _readers
    load_easyocr()
    return _readers


def _preprocess_variants(image_path: str) -> list[np.ndarray]:
    """Return up to 4 image variants that improve OCR on different content types.

    Variant 0 — original (possibly upscaled to ≥600px short side)
    Variant 1 — CLAHE on L-channel (memes with gradient/dark-overlay backgrounds)
    Variant 2 — adaptive threshold (stylized/outlined fonts on busy backgrounds)
    Variant 3 — unsharp mask (blurry or compressed images)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return []

    h, w = img_bgr.shape[:2]
    if min(h, w) < 600:
        scale = 600 / min(h, w)
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    variants: list[np.ndarray] = [img_bgr]

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
    variants.append(cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=8,
    )
    variants.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))

    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=2)
    variants.append(cv2.addWeighted(img_bgr, 1.5, blurred, -0.5, 0))

    return variants


def _run_reader_on_array(reader: Any, img: np.ndarray) -> list[str]:
    try:
        results = reader.readtext(img, detail=0, paragraph=True)
        return [str(r) for r in results if r and str(r).strip()]
    except Exception:
        logger.debug("EasyOCR pass failed on variant", exc_info=True)
        return []


def run_easyocr(image_path: str) -> list[str]:
    """Run all Indic EasyOCR readers with preprocessing. Returns fragments or [].

    Primary reader (en+hi) runs on all 4 preprocessing variants.
    Additional readers run on the original image only to limit latency.
    Does not raise — errors are logged and an empty list is returned.
    """
    readers = _get_readers()
    if not readers:
        return []

    variants = _preprocess_variants(image_path)
    fragments: list[str] = []

    for reader in readers:
        if variants:
            run_on = variants if reader is readers[0] else variants[:1]
            for variant in run_on:
                fragments.extend(_run_reader_on_array(reader, variant))
        else:
            # cv2 could not open the file — try raw path for the primary reader only
            if reader is readers[0]:
                try:
                    results = reader.readtext(image_path, detail=0, paragraph=True)
                    fragments.extend(str(r) for r in results if r and str(r).strip())
                except Exception:
                    logger.debug("EasyOCR raw-path fallback failed", exc_info=True)

    logger.info(
        "EasyOCR extracted %d fragments from %s",
        len(fragments),
        Path(image_path).name,
    )
    return fragments


def is_available() -> bool:
    """Return True if at least one EasyOCR reader is loaded."""
    return bool(_readers)
