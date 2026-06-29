"""Surya-only OCR router."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from backend.pipeline.surya_ocr import load_surya, run_surya_ocr

logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")

SPAM_PHRASES = (
    "click here",
    "subscribe now",
    "follow for more",
    "limited time offer",
    "dm me",
    "whatsapp me",
    "join my channel",
)
SCAM_PHRASES = (
    "free money",
    "guaranteed income",
    "earn money instantly",
    "double your money",
    "risk free profit",
    "investment scheme",
)
PROMOTIONAL_PHRASES = (
    "buy now",
    "sale",
    "discount",
    "sponsored",
    "advertisement",
    "promo code",
    "brand deal",
)
FAKE_HISTORY_PHRASES = (
    "aliens built this temple",
    "secret history they hide",
    "proof historians lied",
    "fake history exposed",
    "ancient astronauts built",
)


def _normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def _merge_fragments(fragments: list[str]) -> str:
    seen: set[str] = set()
    merged: list[str] = []
    for fragment in fragments:
        norm = fragment.strip()
        key = norm.lower()
        if norm and key not in seen:
            seen.add(key)
            merged.append(norm)
    return _normalize_text(" ".join(merged))


def extract_ocr_text(image_path: str) -> str:
    """Extract OCR text with Surya only."""
    logger.info("Surya OCR inference started: %s", Path(image_path).name)
    try:
        fragments = run_surya_ocr(image_path)
        text = _merge_fragments(fragments)
    except Exception:
        logger.exception("Surya OCR inference failed")
        return ""

    logger.info("Surya OCR completed: %d chars", len(text))
    return text


def get_text_quality_score(ocr_text: str, caption: str | None = None) -> float:
    """Return a conservative spam/scam/promotional quality risk score in [0, 1]."""
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


def _get_ocr() -> bool:
    """Warm Surya OCR for legacy callers."""
    return load_surya()
