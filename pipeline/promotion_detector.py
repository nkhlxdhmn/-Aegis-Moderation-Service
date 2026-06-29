"""Promotion and advertising signal fusion for moderation.

Combines OCR/caption text, OpenCLIP similarities, and YOLO detections.
Also detects phone numbers, social media handles, URLs, and QR code content.
Does not make moderation decisions.

Covers English + common Hindi/Hinglish promotional phrases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

WHITESPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(
    r"https?://|www\.|t\.me/|wa\.me/|bit\.ly|tinyurl|short\.url|link\.tree|linktr\.ee",
    re.IGNORECASE,
)
HASHTAG_PATTERN = re.compile(r"#\w+")
# Indian phone numbers: 10-digit starting with 6-9, optional +91 / 0 prefix
PHONE_PATTERN = re.compile(
    r"(?:(?:\+91|0091|0)[\s\-]?)?[6-9]\d{9}\b"
    r"|(?:\+91[\s\-]?\d{5}[\s\-]?\d{5})"
)
# @handle detection (Instagram, Twitter, YouTube)
SOCIAL_HANDLE_PATTERN = re.compile(r"@[A-Za-z0-9_.]{3,30}")

ADVERTISING_PHRASES = (
    # Classic ad signals
    "sponsored", "sponsored ad", "advertisement", "paid partnership",
    "brand promotion", "brand deal", "product promotion",
    "buy now", "order now", "shop now", "get now",
    "limited time offer", "limited time only", "offer ends", "last day", "today only",
    "promo code", "coupon code", "use code", "discount", "sale", "sale today",
    "hurry up", "act now", "don't miss", "last chance",
    # Course / training promotion
    "free course", "free class", "free training", "free workshop",
    "free webinar", "free masterclass", "free session", "free batch",
    "paid course", "paid class", "paid training", "paid batch",
    "course selling", "online course", "join my course", "join my class",
    "join my batch", "buy my course", "enroll now", "enroll today",
    "register now", "registration open", "seats are limited", "limited seats",
    "batch starting", "batch starts", "next batch", "new batch",
    "morning batch", "evening batch", "weekend batch",
    "certification course", "get certified", "get certificate",
    "online training", "live class", "live session", "masterclass",
    "100% free", "totally free", "absolutely free",
    "click here", "click link", "link in bio", "check link", "visit link",
    "book your seat", "book your slot", "book now",
    # Earn money / MLM
    "earn money", "make money", "earn from home", "work from home",
    "passive income", "earn ₹", "earn rs", "investment opportunity",
    "double your money", "risk free", "no risk", "instant profit",
    "100% guaranteed", "make money fast", "refer and earn",
    # Recruitment / hiring
    "recruitment campaign", "we are hiring", "apply now", "job opening",
    "hiring now", "vacancy", "walk in interview",
    # Join / sign up
    "join now", "sign up now", "sign up free", "register free",
    "join telegram", "join whatsapp", "join our group", "join our channel",
    "influencer promotion",
    # Hindi/Hinglish promotional phrases
    "muft course", "muft class", "muft training", "muft batch",
    "abhi join karo", "abhi register karo", "link dekhiye",
    "link pe click karo", "course join karo", "batch join karo",
    "paise kamao", "ghar baithe kamao", "ghar se kamao",
    "free mein sikhein", "bilkul free", "muft mein",
    "abhi enroll karo", "seats siimit hain", "jaldi karo",
    "aaj hi join karo", "aaj hi register karo",
)

AFFILIATE_PHRASES = (
    "affiliate", "affiliate link", "affiliate marketing",
    "referral", "referral code", "referral link",
    "use my code", "use code", "commission", "partner link",
    "crypto signal", "trading signal", "binary trading",
)

SOCIAL_MEDIA_PHRASES = (
    "follow us", "follow me", "follow for more",
    "follow for follow", "follow back", "follow karo",
    "f4f", "l4l", "like for like", "follow4follow",
    "like and follow", "follow and like",
    "like and share", "like share subscribe",
    "subscribe now", "subscribe to my youtube", "subscribe karo",
    "join my channel", "join my telegram", "join my whatsapp",
    "telegram channel", "whatsapp group", "whatsapp me",
    "discord server", "dm me", "dm for details", "dm for price",
    "link in bio", "comment below", "tag a friend",
    "gain followers", "get followers", "buy followers",
    "follow my page", "follow my account", "follow my profile",
    "follow my instagram", "follow my insta",
    "hamara follow karo", "channel subscribe karo",
    "instagram follow karo", "youtube subscribe karo",
)

# Marketing keywords that each count as +1 toward marketing_keyword_count
MARKETING_KEYWORDS: frozenset[str] = frozenset({
    "discount", "offer", "sale", "coupon", "promo", "free", "buy", "shop",
    "order", "subscribe", "follow", "register", "enroll", "limited", "urgent",
    "hurry", "deal", "sponsor", "advertisement", "ad", "earn", "money",
    "income", "cashback", "referral", "affiliate", "commission", "reward",
    "bonus", "prize", "winner", "giveaway",
})

# Course-specific phrases — any match = course promotion
COURSE_PROMOTION_PHRASES: tuple[str, ...] = (
    "free course", "paid course", "free class", "paid class",
    "free training", "paid training", "free workshop", "paid workshop",
    "free webinar", "free masterclass", "free batch", "paid batch",
    "join my course", "join my class", "join my batch",
    "buy my course", "enroll now", "enroll today",
    "batch starting", "batch starts", "next batch", "new batch",
    "morning batch", "evening batch", "weekend batch",
    "certification course", "get certified", "seats are limited",
    "limited seats", "registration open",
    "muft course", "muft class", "muft batch", "muft training",
    "abhi enroll karo", "course join karo", "batch join karo",
)

ADVERTISING_PROMPTS = set(ADVERTISING_PHRASES) | {
    "marketing", "business promotion",
    "product advertisement", "course advertisement",
}
AFFILIATE_PROMPTS = set(AFFILIATE_PHRASES) | {
    "affiliate marketing", "referral marketing",
}
SOCIAL_MEDIA_PROMPTS = set(SOCIAL_MEDIA_PHRASES) | {
    "social media promotion", "telegram promotion",
    "whatsapp promotion", "discord promotion",
    "instagram promotion", "youtube promotion",
}

PROMOTION_OBJECT_LABELS = {
    "product", "package", "box", "bottle", "book",
    "cell phone", "laptop", "tv",
}


def _clamp_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text or "").strip().lower()


def _normalize_label(value: Any) -> str:
    return _normalize_text(str(value or "").replace("-", " ").replace("_", " "))


def _phrase_score(text: str, phrases: Sequence[str]) -> float:
    matches = sum(1 for phrase in phrases if phrase in text)
    if matches == 0:
        return 0.0
    return _clamp_score(0.35 + min(matches - 1, 3) * 0.15)


def _text_scores(ocr_text: str | None, caption: str | None) -> tuple[float, float, float]:
    combined = _normalize_text(f"{ocr_text or ''} {caption or ''}")
    if not combined:
        return 0.0, 0.0, 0.0

    advertising_score  = _phrase_score(combined, ADVERTISING_PHRASES)
    affiliate_score    = _phrase_score(combined, AFFILIATE_PHRASES)
    social_media_score = _phrase_score(combined, SOCIAL_MEDIA_PHRASES)

    url_count     = len(URL_PATTERN.findall(combined))
    hashtag_count = len(HASHTAG_PATTERN.findall(combined))
    if url_count:
        advertising_score = max(advertising_score, min(0.25 + url_count * 0.10, 0.65))
    if hashtag_count >= 6:
        social_media_score = max(social_media_score, 0.35)

    return (
        _clamp_score(advertising_score),
        _clamp_score(affiliate_score),
        _clamp_score(social_media_score),
    )


def _extra_signals(
    ocr_text: str | None, caption: str | None, qr_decoded_text: str | None
) -> dict[str, float]:
    """Detect phone numbers, social handles, URLs, and marketing keyword density."""
    combined = _normalize_text(f"{ocr_text or ''} {caption or ''} {qr_decoded_text or ''}")
    raw_ocr  = f"{ocr_text or ''} {caption or ''}"

    # Phone numbers
    phones = PHONE_PATTERN.findall(raw_ocr)
    phone_score = _clamp_score(min(len(phones) * 0.40, 0.80)) if phones else 0.0

    # Social handles (@username)
    handles = SOCIAL_HANDLE_PATTERN.findall(raw_ocr)
    handle_score = _clamp_score(min(len(handles) * 0.30, 0.75)) if handles else 0.0

    # URLs
    urls = URL_PATTERN.findall(combined)
    url_score = _clamp_score(min(len(urls) * 0.35, 0.80)) if urls else 0.0

    # Marketing keyword density
    tokens = set(combined.split())
    kw_count = sum(1 for w in tokens if w in MARKETING_KEYWORDS)

    # Course promotion score
    course_score = _clamp_score(
        _phrase_score(combined, COURSE_PROMOTION_PHRASES) if combined else 0.0
    )

    return {
        "phone_number_score":    phone_score,
        "social_handle_score":   handle_score,
        "url_score":             url_score,
        "marketing_keyword_count": float(kw_count),
        "course_promotion_score":  course_score,
    }


def _iter_clip_items(openclip_similarities: Any) -> list[tuple[str, float]]:
    if openclip_similarities is None:
        return []
    if isinstance(openclip_similarities, Mapping):
        source = openclip_similarities
        for key in ("promotion_scores", "prompt_scores", "similarities", "scores"):
            nested = openclip_similarities.get(key)
            if isinstance(nested, Mapping):
                source = nested
                break
        return [(_normalize_label(k), _clamp_score(v)) for k, v in source.items()]
    return []


def _clip_scores(openclip_similarities: Any) -> tuple[float, float, float]:
    advertising_terms = {_normalize_label(l) for l in ADVERTISING_PROMPTS}
    affiliate_terms   = {_normalize_label(l) for l in AFFILIATE_PROMPTS}
    social_terms      = {_normalize_label(l) for l in SOCIAL_MEDIA_PROMPTS}

    ad, af, sm = 0.0, 0.0, 0.0
    for label, score in _iter_clip_items(openclip_similarities):
        if label in advertising_terms:
            ad = max(ad, score)
        elif label in affiliate_terms:
            af = max(af, score)
        elif label in social_terms:
            sm = max(sm, score)
    return ad, af, sm


def _yolo_promotion_score(yolo_detections: Sequence[Mapping[str, Any]]) -> float:
    object_terms = {_normalize_label(l) for l in PROMOTION_OBJECT_LABELS}
    score = 0.0
    for det in yolo_detections:
        if not isinstance(det, Mapping):
            continue
        label = _normalize_label(
            det.get("class") or det.get("label") or det.get("name") or ""
        )
        conf  = _clamp_score(det.get("confidence") or det.get("score") or 0.0)
        if label in object_terms:
            score = max(score, conf * 0.25)
    return _clamp_score(score)


def _fuse_scores(*scores: float) -> float:
    strongest = max((_clamp_score(s) for s in scores), default=0.0)
    combined  = 1.0
    for s in scores:
        combined *= 1.0 - _clamp_score(s)
    return _clamp_score(max(strongest, (1.0 - combined) * 0.85))


def analyze_promotion(
    ocr_text: str | None,
    caption: str | None,
    openclip_similarities: Any,
    yolo_detections: Sequence[Mapping[str, Any]] | None,
    qr_decoded_text: str | None = None,
) -> dict[str, float]:
    """Combine OCR, caption, OpenCLIP, YOLO, and QR signals for promotion detection."""
    logger.info("Promotion signal fusion started")

    text_ad, text_af, text_sm = _text_scores(ocr_text, caption)
    clip_ad, clip_af, clip_sm = _clip_scores(openclip_similarities)
    yolo_support = _yolo_promotion_score(yolo_detections or [])
    extra = _extra_signals(ocr_text, caption, qr_decoded_text)

    advertising_score  = _fuse_scores(text_ad, clip_ad, yolo_support)
    affiliate_score    = _fuse_scores(text_af, clip_af)
    social_media_score = _fuse_scores(text_sm, clip_sm)
    promotion_score    = _clamp_score(
        max(advertising_score, affiliate_score, social_media_score)
    )

    logger.info("Promotion signal fusion completed")
    return {
        "promotion_score":         promotion_score,
        "advertising_score":       advertising_score,
        "affiliate_score":         affiliate_score,
        "social_media_score":      social_media_score,
        **extra,
    }


# Backward-compat alias
get_promotion_scores = analyze_promotion
