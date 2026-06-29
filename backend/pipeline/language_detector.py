"""Language detection for multilingual OCR text.

Uses langdetect (rule-based, no GPU, ~500 KB) to identify the primary script/
language of extracted OCR text.  Result is informational — used for logging,
audit trails, and routing decisions.

Install: pip install langdetect
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Language metadata ─────────────────────────────────────────────────────────

LANG_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "ur": "Urdu",
    "as": "Assamese",
    "sa": "Sanskrit",
    "ne": "Nepali",
    "or": "Odia",
    "ml": "Malayalam",
    "sd": "Sindhi",
    "ar": "Arabic",
}

INDIC_LANGS: frozenset[str] = frozenset(
    {
        "hi",
        "mr",
        "ta",
        "te",
        "kn",
        "bn",
        "pa",
        "gu",
        "ur",
        "as",
        "sa",
        "ne",
        "or",
        "ml",
        "sd",
    }
)

# Unicode ranges used as a fast script-detection fallback when langdetect fails
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_ARABIC = re.compile(r"[؀-ۿ]")  # covers Urdu
_BENGALI = re.compile(r"[ঀ-৿]")
_TAMIL = re.compile(r"[஀-௿]")
_TELUGU = re.compile(r"[ఀ-౿]")
_KANNADA = re.compile(r"[ಀ-೿]")
_GUJARATI = re.compile(r"[઀-૿]")
_GURMUKHI = re.compile(r"[਀-੿]")  # Punjabi


def _unicode_fallback(text: str) -> str | None:
    """Guess script family from Unicode codepoints when langdetect is unavailable."""
    counts = {
        "hi": len(_DEVANAGARI.findall(text)),
        "ur": len(_ARABIC.findall(text)),
        "bn": len(_BENGALI.findall(text)),
        "ta": len(_TAMIL.findall(text)),
        "te": len(_TELUGU.findall(text)),
        "kn": len(_KANNADA.findall(text)),
        "gu": len(_GUJARATI.findall(text)),
        "pa": len(_GURMUKHI.findall(text)),
    }
    best_lang, best_count = max(counts.items(), key=lambda x: x[1])
    return best_lang if best_count > 3 else None


def detect(text: str | None) -> str:
    """Return the BCP-47 language code of *text*.

    Falls back to Unicode script detection, then 'en' if both fail.
    Requires at least 10 characters for reliable detection.
    """
    if not text or len(text.strip()) < 10:
        return "en"

    try:
        from langdetect import detect as _detect

        lang = _detect(text)
        logger.debug("langdetect: %s", lang)
        return lang
    except Exception:
        pass

    fb = _unicode_fallback(text)
    if fb:
        logger.debug("unicode fallback: %s", fb)
        return fb

    return "en"


def is_indic(lang_code: str) -> bool:
    """True if *lang_code* is a supported Indic language."""
    return lang_code in INDIC_LANGS


def lang_name(lang_code: str) -> str:
    """Human-readable name for a BCP-47 code, e.g. 'hi' → 'Hindi'."""
    return LANG_NAMES.get(lang_code, lang_code.upper())
