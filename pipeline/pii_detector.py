"""Privacy and PII detection signals.

This module uses regex-based Indian PII detection with optional Presidio support
when the package is available. It does not make moderation decisions.
"""

from __future__ import annotations

from functools import lru_cache
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

WHITESPACE_PATTERN = re.compile(r"\s+")

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
AADHAAR_PATTERN = re.compile(r"(?<!\d)(?:\d{4}[\s-]?){2}\d{4}(?!\d)")
PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
PASSPORT_PATTERN = re.compile(r"\b[A-Z][0-9]{7}\b", re.IGNORECASE)
BANK_ACCOUNT_PATTERN = re.compile(r"(?<!\d)\d{9,18}(?!\d)")
UPI_PATTERN = re.compile(r"\b[A-Z0-9._-]{2,}@[A-Z]{2,}\b", re.IGNORECASE)
IFSC_PATTERN = re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE)
ADDRESS_PATTERN = re.compile(
    r"\b(?:house|flat|apt|apartment|door|plot|street|road|lane|nagar|colony|"
    r"sector|village|district|pin|pincode|postal code)\b",
    re.IGNORECASE,
)

PRESIDIO_ENTITY_MAP = {
    "EMAIL_ADDRESS": "email_detected",
    "PHONE_NUMBER": "phone_detected",
    "IN_AADHAAR": "aadhaar_detected",
    "IN_PAN": "pan_detected",
    "IN_PASSPORT": "passport_detected",
    "IBAN_CODE": "bank_info_detected",
    "CREDIT_CARD": "bank_info_detected",
    "LOCATION": "address_detected",
}


def _normalize_text(text: str | None) -> str:
    """Normalize OCR/caption text for PII matching."""

    return WHITESPACE_PATTERN.sub(" ", text or "").strip()


def _digits_only(value: str) -> str:
    """Return only digits from a value."""

    return re.sub(r"\D", "", value)


def _valid_aadhaar_candidate(value: str) -> bool:
    """Return whether a regex match looks like an Aadhaar number."""

    digits = _digits_only(value)
    if len(digits) != 12:
        return False
    if digits[0] in {"0", "1"}:
        return False
    return len(set(digits)) > 1


def _has_aadhaar(text: str) -> bool:
    """Detect Aadhaar numbers with light false-positive filtering."""

    return any(
        _valid_aadhaar_candidate(match.group(0))
        for match in AADHAAR_PATTERN.finditer(text)
    )


def _has_bank_account(text: str) -> bool:
    """Detect bank account-like numbers without double-counting Aadhaar."""

    for match in BANK_ACCOUNT_PATTERN.finditer(text):
        value = match.group(0)
        if _valid_aadhaar_candidate(value):
            continue
        return True
    return False


def _has_address(text: str) -> bool:
    """Detect personal-address-like text."""

    return bool(ADDRESS_PATTERN.search(text)) and bool(re.search(r"\d", text))


@lru_cache(maxsize=1)
def _get_presidio_analyzer() -> Any | None:
    """Return a cached Presidio analyzer when available."""

    try:
        from presidio_analyzer import AnalyzerEngine

        return AnalyzerEngine()
    except Exception:
        logger.info("Presidio is unavailable; using regex-only PII detection")
        return None


def _presidio_flags(text: str) -> dict[str, bool]:
    """Return detector flags inferred from Presidio, if available."""

    analyzer = _get_presidio_analyzer()
    flags = {
        "aadhaar_detected": False,
        "pan_detected": False,
        "passport_detected": False,
        "phone_detected": False,
        "email_detected": False,
        "bank_info_detected": False,
        "address_detected": False,
    }
    if analyzer is None or not text:
        return flags

    try:
        results = analyzer.analyze(text=text, language="en")
    except Exception:
        logger.exception("Presidio PII analysis failed; using regex-only result")
        return flags

    for result in results:
        entity_type = getattr(result, "entity_type", "")
        flag_name = PRESIDIO_ENTITY_MAP.get(entity_type)
        if flag_name:
            flags[flag_name] = True

    return flags


def _regex_flags(text: str) -> dict[str, bool]:
    """Return detector flags inferred from regex patterns."""

    return {
        "aadhaar_detected": _has_aadhaar(text),
        "pan_detected": bool(PAN_PATTERN.search(text)),
        "passport_detected": bool(PASSPORT_PATTERN.search(text)),
        "phone_detected": bool(PHONE_PATTERN.search(text)),
        "email_detected": bool(EMAIL_PATTERN.search(text)),
        "bank_info_detected": bool(
            UPI_PATTERN.search(text)
            or IFSC_PATTERN.search(text)
            or _has_bank_account(text)
        ),
        "address_detected": _has_address(text),
    }


def _pii_score(flags: dict[str, bool]) -> float:
    """Return an aggregate PII exposure score."""

    weights = {
        "aadhaar_detected": 0.55,
        "pan_detected": 0.45,
        "passport_detected": 0.45,
        "phone_detected": 0.30,
        "email_detected": 0.25,
        "bank_info_detected": 0.50,
        "address_detected": 0.35,
    }
    score = sum(weight for key, weight in weights.items() if flags.get(key))
    return max(0.0, min(1.0, score))


def analyze_pii(ocr_text: str | None, caption: str | None) -> dict[str, Any]:
    """Return PII detection signals for OCR text and caption."""

    logger.info("PII analysis started")
    text = _normalize_text(f"{ocr_text or ''} {caption or ''}")
    regex_flags = _regex_flags(text)
    presidio_flags = _presidio_flags(text)
    combined_flags = {
        key: bool(regex_flags.get(key) or presidio_flags.get(key))
        for key in regex_flags
    }

    result = {
        "pii_score": _pii_score(combined_flags),
        "aadhaar_detected": combined_flags["aadhaar_detected"],
        "pan_detected": combined_flags["pan_detected"],
        "passport_detected": combined_flags["passport_detected"],
        "phone_detected": combined_flags["phone_detected"],
        "email_detected": combined_flags["email_detected"],
        "bank_info_detected": combined_flags["bank_info_detected"],
    }
    logger.info("PII analysis completed")
    return result


def get_pii_scores(ocr_text: str | None, caption: str | None) -> dict[str, Any]:
    """Compatibility alias for PII analysis."""

    return analyze_pii(ocr_text, caption)
