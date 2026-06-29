"""Zero-tolerance hard-block filter.

Runs immediately after OCR — before any GPU/ML inference.
Any match → content is REJECTED regardless of vision scores.

Covers English and common Hindi/Hinglish transliterations.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Phrases: no legitimate use on a cultural/heritage platform ────────────────

HARD_BLOCK_PHRASES: frozenset[str] = frozenset(
    {
        # ── Social media follow/subscribe solicitation ───────────────────────────
        "follow us",
        "follow me",
        "follow for more",
        "follow for follow",
        "follow back",
        "follow my page",
        "follow our page",
        "follow my account",
        "follow my profile",
        "follow my instagram",
        "follow my insta",
        "follow karo",
        "hamara follow karo",
        "follow kar",
        "subscribe",
        "subscribe now",
        "subscribe to",
        "subscribe karo",
        "subscribe kar",
        "like and subscribe",
        "like share subscribe",
        "join now",
        "join our",
        "join my",
        "join telegram",
        "join whatsapp",
        "join channel",
        "join group",
        "join karo",
        "telegram channel",
        "telegram group",
        "whatsapp group",
        "whatsapp channel",
        "whatsapp me",
        "discord server",
        "link in bio",
        "link in story",
        "check our link",
        "check link",
        "dm me",
        "dm for",
        "dm for details",
        "dm for price",
        "message me",
        "inbox me",
        # ── Commercial promotion ──────────────────────────────────────────────────
        "discount",
        "coupon code",
        "promo code",
        "use code",
        "buy now",
        "shop now",
        "order now",
        "purchase now",
        "limited offer",
        "limited time offer",
        "offer ends",
        "sale today",
        "today only",
        "50% off",
        "40% off",
        "30% off",
        "20% off",
        "flat off",
        "upto off",
        "free course",
        "free class",
        "free training",
        "free workshop",
        "free webinar",
        "free masterclass",
        "free batch",
        "free session",
        "paid course",
        "paid class",
        "paid training",
        "paid batch",
        "enroll now",
        "enroll today",
        "register now",
        "registration open",
        "register free",
        "book now",
        "book your seat",
        "book your slot",
        "100% free",
        "totally free",
        "absolutely free",
        "earn money",
        "make money",
        "earn from home",
        "work from home",
        "passive income",
        "refer and earn",
        "batch starting",
        "batch starts",
        "next batch",
        "new batch",
        "morning batch",
        "evening batch",
        "weekend batch",
        "get certified",
        "certification course",
        "seats are limited",
        "limited seats",
        # Hindi transliterations of promotional phrases
        "muft course",
        "muft class",
        "muft training",
        "abhi register karo",
        "abhi join karo",
        "link dekhiye",
        "link pe click karo",
        "course join karo",
        "batch join karo",
        "paise kamao",
        "ghar baithe kamao",
        # ── Political campaigns ───────────────────────────────────────────────────
        "vote for",
        "vote now",
        "cast your vote",
        "elect",
        "election campaign",
        "political rally",
        "support party",
        "campaign for",
        "candidate for",
        "vote against",
        "vote karein",
        "vote karo",
        "apna vote do",
        # ── Violence / hatred (English) ───────────────────────────────────────────
        "kill muslims",
        "kill hindus",
        "kill jews",
        "kill christians",
        "kill sikhs",
        "kill dalits",
        "kill brahmins",
        "death to muslims",
        "death to hindus",
        "death to jews",
        "death to christians",
        "death to sikhs",
        "hang muslims",
        "hang hindus",
        "hang them",
        "lynch them",
        "lynch muslims",
        "lynch hindus",
        "boycott muslims",
        "boycott hindus",
        "boycott christians",
        "boycott sikhs",
        "boycott jews",
        "expel muslims",
        "expel hindus",
        "expel christians",
        "ethnic cleansing",
        "genocide",
        "exterminate",
        "white supremacy",
        "hindu supremacy",
        "islamic supremacy",
        "religious war",
        "holy war against",
        "gas them",
        "burn them all",
        "rape them",
        "go back to your country",
        # ── Divisive propaganda ───────────────────────────────────────────────────
        "muslims are enemies",
        "hindus are enemies",
        "christians are enemies",
        "sikhs are enemies",
        "dalits are enemies",
        "traitors must die",
        "invaders go back",
        "cow eaters get out",
        "idol worshippers get out",
        # ── Abusive / self-harm instigation ──────────────────────────────────────
        "kill yourself",
        "go die",
        "i will kill you",
        "i will hurt you",
        "you deserve to die",
    }
)

# Single words that ALWAYS block regardless of context on this platform
HARD_BLOCK_SINGLE_WORDS: frozenset[str] = frozenset(
    {
        "genocide",
        "supremacy",
        "exterminate",
    }
)

# These words block only when appearing within 3 tokens of a community name
VIOLENCE_WORDS: frozenset[str] = frozenset(
    {
        "boycott",
        "expel",
        "eliminate",
        "destroy",
        "kill",
        "hang",
        "lynch",
    }
)
COMMUNITY_WORDS: frozenset[str] = frozenset(
    {
        "muslim",
        "muslims",
        "hindu",
        "hindus",
        "christian",
        "christians",
        "sikh",
        "sikhs",
        "jewish",
        "jews",
        "dalit",
        "dalits",
        "brahmin",
        "brahmins",
        "black",
        "white",
        "asian",
        "arab",
    }
)

_TOKEN_PROXIMITY = 4  # max token gap for violence+community pair


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _check_proximity(tokens: list[str]) -> bool:
    """True if a VIOLENCE_WORD appears within _TOKEN_PROXIMITY tokens of a COMMUNITY_WORD."""
    for i, tok in enumerate(tokens):
        if tok in VIOLENCE_WORDS:
            window = tokens[max(0, i - _TOKEN_PROXIMITY) : i + _TOKEN_PROXIMITY + 1]
            if any(w in COMMUNITY_WORDS for w in window):
                return True
    return False


def check(text: str | None) -> tuple[bool, str]:
    """Return (blocked, reason).

    blocked=True means the content must be REJECTED immediately.
    Runs in microseconds — no model loading.
    """
    if not text:
        return False, ""

    normalized = _normalize(text)
    tokens = normalized.split()

    # Phrase match
    for phrase in HARD_BLOCK_PHRASES:
        if phrase in normalized:
            logger.warning("Hard-block phrase match: %r", phrase)
            return True, f"Hard-block phrase: '{phrase}'"

    # Single-word match
    token_set = set(tokens)
    for word in HARD_BLOCK_SINGLE_WORDS:
        if word in token_set:
            logger.warning("Hard-block single word: %r", word)
            return True, f"Hard-block word: '{word}'"

    # Proximity check: violence word near community word
    if _check_proximity(tokens):
        logger.warning("Hard-block proximity: violence+community detected")
        return True, "Violence targeting a community detected"

    return False, ""
