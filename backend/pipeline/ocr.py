"""OCR router â€” Surya primary, EasyOCR fallback.

Routing logic:
  1. Try Surya OCR first (pipeline/surya_ocr.py).
  2. If Surya returns fragments â†’ use them exclusively, return.
  3. If Surya returns nothing (unavailable, failure, or blank image) â†’
     run EasyOCR (pipeline/easyocr_engine.py) and return its result.
  4. Never run both engines on the same image.

Public API (unchanged for all callers):
  extract_ocr_text(image_path)          -> str
  get_text_quality_score(text, caption) -> float [0, 1]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from backend.pipeline.surya_ocr import run_surya_ocr
from backend.pipeline.easyocr_engine import run_easyocr, _get_readers

logger = logging.getLogger(__name__)

# â”€â”€ Regex helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")

SPAM_PHRASES = (
    "click here", "subscribe now", "follow for more",
    "limited time offer", "dm me", "whatsapp me", "join my channel",
)
SCAM_PHRASES = (
    "free money", "guaranteed income", "earn money instantly",
    "double your money", "risk free profit", "investment scheme",
)
PROMOTIONAL_PHRASES = (
    "buy now", "sale", "discount", "sponsored",
    "advertisement", "promo code", "brand deal",
)
FAKE_HISTORY_PHRASES = (
    "aliens built this temple", "secret history they hide",
    "proof historians lied", "fake history exposed",
    "ancient astronauts built",
)


def _normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def _merge_fragments(*fragment_lists: list[str]) -> str:
    seen: set[str] = set()
    merged: list[str] = []
    for fragments in fragment_lists:
        for fragment in fragments:
            norm = fragment.strip()
            key = norm.lower()
            if norm and key not in seen:
                seen.add(key)
                merged.append(norm)
    return _normalize_text(" ".join(merged))


# â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def extract_ocr_text(image_path: str) -> str:
    """Extract multilingual OCR text â€” Surya primary, EasyOCR fallback.

    Returns an empty string on any unrecoverable error.
    """
    logger.info("OCR inference started: %s", Path(image_path).name)
    try:
        surya_fragments = run_surya_ocr(image_path)
        if surya_fragments:
            logger.info("Surya OCR pass: %d fragments (primary succeeded)", len(surya_fragments))
            text = _merge_fragments(surya_fragments)
        else:
            logger.info("Surya OCR produced no text â€” running EasyOCR fallback")
            easyocr_fragments = run_easyocr(image_path)
            if easyocr_fragments:
                logger.info("EasyOCR fallback: %d fragments", len(easyocr_fragments))
            text = _merge_fragments(easyocr_fragments)
    except Exception:
        logger.exception("OCR inference failed")
        return ""

    logger.info("OCR completed: %d chars", len(text))
    return text


def get_text_quality_score(ocr_text: str, caption: str | None = None) -> float:
    """Return a conservative spam/scam/promotional quality risk score [0, 1]."""
    logger.info("Text quality scoring started")
    try:
        combined = _normalize_text(f"{ocr_text or ''} {caption or ''}").lower()
        if not combined:
            return 0.0

        score = 0.0
        url_count = len(URL_PATTERN.findall(combined))
        if url_count >= 3:
            score += 0.4
        elif url_count >= 1:
            score += 0.2

        phrase_groups = (
            (SPAM_PHRASES, 0.2),
            (SCAM_PHRASES, 0.35),
            (PROMOTIONAL_PHRASES, 0.2),
            (FAKE_HISTORY_PHRASES, 0.35),
        )
        for phrases, weight in phrase_groups:
            matches = sum(1 for phrase in phrases if phrase in combined)
            if matches:
                score += min(weight * matches, weight + 0.15)

        if combined.count("#") >= 8:
            score += 0.2

        final_score = max(0.0, min(1.0, score))
    except Exception:
        logger.exception("Text quality scoring failed")
        return 0.0

    logger.info("Text quality scoring completed")
    return final_score


# â”€â”€ Backward-compatibility shims â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# model_warmup.py calls ocr._get_ocr(); validate_models.py calls _get_readers().
# Both delegate to easyocr_engine so callers see the live reader list.

def _get_ocr() -> list:
    """Alias for model_warmup.py â€” warms/returns EasyOCR readers."""
    return _get_readers()
