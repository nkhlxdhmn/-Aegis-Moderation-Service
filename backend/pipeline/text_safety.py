"""Rule-based text safety signals — hate speech, political, terrorism, fraud.

Covers English + common Hindi/Urdu transliterations used in Indian social media.
Does not call external APIs or make moderation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)

WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class TextRule:
    phrases: tuple[str, ...]
    weight: float = 0.35


RULES: dict[str, tuple[TextRule, ...]] = {

    # ── Terrorism / extremism ────────────────────────────────────────────────
    "terrorism_score": (
        TextRule((
            "terrorism promotion", "join terrorist", "support terrorism",
            "extremist propaganda", "extremism", "radicalize",
            "join isis", "join al qaeda", "join lashkar", "join jaish",
            "join naxal", "join maoist", "join khalistani",
            "jihad against india", "holy war against india",
            "bomb blast", "suicide bomber", "ied attack",
            "stone pelting", "urban naxal",
        ), weight=0.45),
        TextRule((
            "bomb making instructions", "how to make a bomb",
            "homemade explosive", "weapon construction instructions",
            "build a gun", "make a firearm",
        ), weight=0.50),
        TextRule((
            "drug trafficking", "sell illegal drugs",
            "human trafficking", "trafficking victims",
        ), weight=0.40),
    ),

    # ── Fraud / scam ─────────────────────────────────────────────────────────
    "fraud_score": (
        TextRule((
            "phishing", "verify your password", "account will be suspended",
            "login to claim", "send otp", "enter otp here",
            "your account is hacked", "urgent bank verification",
        ), weight=0.40),
        TextRule((
            "scam", "guaranteed income", "double your money",
            "fake investment", "investment scheme", "risk free profit",
            "crypto giveaway", "fake giveaway", "financial fraud",
            "ponzi scheme", "mlm scheme", "pyramid scheme",
        ), weight=0.45),
    ),

    # ── Hate speech (English) ────────────────────────────────────────────────
    "hate_speech_score": (
        # Direct community attack
        TextRule((
            "kill all muslims", "kill all hindus", "kill all christians",
            "kill all sikhs", "kill all jews", "kill all dalits",
            "death to muslims", "death to hindus", "death to christians",
            "death to sikhs", "death to jews",
            "gas the muslims", "gas the hindus", "gas the jews",
            "hang the muslims", "hang the hindus",
            "lynch muslims", "lynch hindus",
            "rape muslim women", "rape hindu women",
        ), weight=0.90),
        # Dehumanisation
        TextRule((
            "muslims are not human", "hindus are not human",
            "christians are animals", "sikhs are animals",
            "dalits are untouchables", "dalits are sub-human",
            "schedule caste scum", "obc filth",
            "go back to your country", "invaders get out",
            "cow eaters get out", "idol worshippers get out",
            "beef eaters are filth",
        ), weight=0.80),
        # Supremacy / ideology
        TextRule((
            "white supremacy", "white power", "hindu rashtra for hindus only",
            "islamic caliphate", "religious supremacy",
            "ethnic cleansing", "genocide", "exterminate",
            "final solution", "religious war",
            "kafirs must die", "kafir", "kuffar must die",
        ), weight=0.85),
        # Religious hatred (English)
        TextRule((
            "religion based attack", "attack this religion",
            "caste based attack", "ethnicity based attack",
            "race based attack", "nationality based attack",
            "all muslims are terrorists", "all hindus are extremists",
            "all christians are colonizers",
            "hate all hindus", "hate all muslims", "hate all christians",
            "lower caste filth", "dirty caste", "dirty race",
            "ethnic scum",
        ), weight=0.60),
        # Hindi/Urdu transliterations of hate speech
        TextRule((
            "musalmanon ko maro", "hinduon ko maro", "sikkhon ko maro",
            "musalman haramkhor", "hinduon ki maa ki", "kafir ko maro",
            "dalit ko maro", "bhangi", "chamar gali",
            "musalman desh se nikalo", "hindu desh se nikalo",
            "pakistan zindabad india murdabad",
            "bharat mata ki jai nahi bolenge toh",
        ), weight=0.70),
        # Divisive propaganda
        TextRule((
            "muslims are enemies", "hindus are enemies",
            "christians are enemies", "sikhs are enemies",
            "dalits are enemies",
            "traitors must die", "invaders go back",
            "hindus are outsiders", "muslims are outsiders",
            "one community one nation exclude",
        ), weight=0.65),
    ),

    # ── Harassment / threats ─────────────────────────────────────────────────
    "harassment_score": (
        TextRule((
            "bullying", "targeted abuse", "harass them",
            "humiliate them", "everyone mock",
            "post her photos", "expose him", "leak her number",
        ), weight=0.40),
        TextRule((
            "i will kill you", "kill yourself", "i will hurt you",
            "threaten you", "you deserve to be beaten",
            "i know where you live", "you will regret this",
            "i will find you",
        ), weight=0.50),
    ),

    # ── Misinformation ───────────────────────────────────────────────────────
    "misinformation_score": (
        TextRule((
            "medical misinformation", "vaccines cause infertility",
            "cure cancer instantly", "drink bleach", "miracle cure",
            "covid is fake", "vaccine is poison", "5g causes covid",
        ), weight=0.40),
        TextRule((
            "fake government notice", "fake court order",
            "official government order", "election manipulation",
            "vote twice", "wrong voting date", "polling station changed",
            "eci is corrupt", "election is rigged",
        ), weight=0.45),
    ),

    # ── Self-harm instigation ─────────────────────────────────────────────────
    "self_harm_text_score": (
        TextRule((
            "suicide encouragement", "you should commit suicide",
            "kill yourself", "end your life",
            "self harm encouragement", "cut yourself", "hurt yourself",
            "nobody cares if you die",
        ), weight=0.50),
    ),

    # ── Political campaigns ───────────────────────────────────────────────────
    "political_score": (
        TextRule((
            "vote for", "vote now", "cast your vote",
            "elect", "election campaign",
            "political rally", "political advertisement",
            "support party", "campaign for",
            "candidate for", "vote against",
            "vote karein", "vote karo", "apna vote do",
            "our candidate", "our party will",
            "abki baar", "phir ek baar",
        ), weight=0.60),
        TextRule((
            "political propaganda", "political mobilization",
            "party promotion", "bjp campaign", "congress campaign",
            "aap campaign", "sp campaign", "bsp campaign",
            "vote bank", "minority appeasement", "majority appeasement",
        ), weight=0.55),
        # Common Indian political slogans that appear in campaign material
        TextRule((
            "jai bharat", "vote for change", "change the government",
            "remove the government", "overthrow the government",
            "political revolution", "vote for development",
        ), weight=0.40),
    ),

    # ── Political campaign (more specific — higher confidence) ───────────────
    "political_campaign_score": (
        TextRule((
            "vote for", "election campaign", "political rally",
            "support the party", "vote on", "polling day",
            "vote for candidate", "vote for party",
            "your vote matters", "voting booth",
            "abhi vote karo", "apna neta chuniye",
        ), weight=0.70),
        TextRule((
            "political advertisement", "political poster",
            "sponsored by party", "paid political",
            "authorized political communication",
        ), weight=0.75),
    ),
}

OUTPUT_KEYS = (
    "terrorism_score",
    "fraud_score",
    "hate_speech_score",
    "harassment_score",
    "misinformation_score",
    "self_harm_text_score",
    "political_score",
    "political_campaign_score",
)


def _normalize_text(text: str | None) -> str:
    return WHITESPACE_PATTERN.sub(" ", text or "").strip().lower()


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _rule_score(text: str, rules: tuple[TextRule, ...]) -> float:
    score = 0.0
    for rule in rules:
        matches = sum(1 for phrase in rule.phrases if phrase in text)
        if matches:
            score += min(rule.weight + (matches - 1) * 0.10, rule.weight + 0.30)
    return _clamp_score(score)


def analyze_text_safety(ocr_text: str | None, caption: str | None) -> dict[str, float]:
    """Return text safety scores for OCR text and caption."""
    logger.info("Advanced text safety analysis started")
    combined = _normalize_text(f"{ocr_text or ''} {caption or ''}")
    if not combined:
        return {key: 0.0 for key in OUTPUT_KEYS}

    scores = {key: _rule_score(combined, RULES.get(key, ())) for key in OUTPUT_KEYS}
    logger.info("Advanced text safety analysis completed")
    return scores


# Backward-compat alias
get_text_safety_scores = analyze_text_safety
