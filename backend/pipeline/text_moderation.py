"""Text post moderation pipeline for Aegis.

Models (all lazy-loaded, each falls back gracefully if unavailable):
  Detoxify multilingual â€” ML toxicity across 7 dimensions (obscene, threat,
      insult, identity_attack, sexual_explicit, toxicity, severe_toxicity).
  FastText lid.176.bin â€” language identification.
  Llama-3.1-8B-AWQ â€” structured JSON reasoning, shared singleton with the
      image pipeline via vlm_engine.reason_text_moderation().

Produces a scores dict fully compatible with decision_engine.decide_with_reason_code().

Text-specific score keys (new keys used by Tier 0 rules in decision_engine.py):
  marketing_keyword_count   â€” phrase count (â‰¥ 2 â†’ Tier-0C reject)
  course_promotion_score    â€” 0-1 course / paid-class marketing
  political_score           â€” 0-1 political content probability
  political_campaign_score  â€” 0-1 active campaign / vote solicitation
  animal_cruelty_text_score â€” 0-1 animal abuse language
  human_killing_text_score  â€” 0-1 murder / execution language
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

WHITESPACE = re.compile(r"\s+")

# â”€â”€ Model paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Path is relative to this file's directory so it resolves correctly regardless
# of the process working directory (e.g. Docker WORKDIR vs local dev).
import os as _os
FASTTEXT_MODEL_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "models", "lid.176.bin"
)
DETOXIFY_MODEL = "multilingual"              # falls back to "unbiased" on import error

# â”€â”€ Frame-level config (reused for text ensemble) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_MAX_TEXT_LEN = 2000  # characters fed to Detoxify / Llama

# â”€â”€ Marketing / promotion keyword lists â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_MARKETING_PHRASES = (
    "buy now", "limited offer", "discount", "sale today", "join now",
    "click link", "click here", "register now", "earn money", "free course",
    "paid course", "subscribe", "promo code", "investment opportunity",
    "telegram link", "whatsapp group", "dm for details", "100% guaranteed",
    "earn â‚¹", "crypto signal", "make money fast", "work from home",
    "passive income", "refer and earn", "affiliate link", "use my code",
    "double your money", "risk free", "no risk", "instant profit",
)

_COURSE_PHRASES = (
    "buy my course", "join my paid class", "limited seats", "seats are limited",
    "registration open", "enroll now", "dm for course", "paid class",
    "online class registration", "join my batch", "my coaching",
    "batch starting", "course starting", "pay and join",
)

_POLITICAL_PHRASES = (
    "vote for", "support the party", "campaign for", "join the movement",
    "elect", "party promotion", "vote our", "join our party",
    "rally for", "political rally", "support our candidate",
    "cast your vote", "vote in favour", "support this leader",
)

_POLITICAL_TOPIC_PHRASES = (
    "prime minister", "chief minister", "parliament", "election",
    "bjp", "congress", "aap", "political party", "candidate",
    "manifesto", "constituency", "lok sabha", "vidhan sabha",
    "vote bank", "political agenda",
)

_ANIMAL_CRUELTY_PHRASES = (
    "animal killing", "animal torture", "animal abuse", "dead animal",
    "hunting violence", "animal attacked", "injured animal", "animal blood",
    "slaughter animal", "kill the animal", "beat the animal",
    "animal fight", "cockfight", "dog fight",
)

_HUMAN_KILLING_PHRASES = (
    "murder", "execution", "dead body", "human killing", "killed by",
    "person stabbed", "person shot", "was tortured", "beheading",
    "lynching", "mob lynching", "honour killing", "contract killing",
    "shoot to kill", "kill him", "kill her", "death threat",
)

# â”€â”€ Heritage keyword detection for text posts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_HERITAGE_KEYWORDS = (
    "ramayana", "mahabharata", "bhagavad gita", "gita", "vedas", "vedic",
    "upanishad", "purana", "krishna", "rama", "vishnu", "shiva", "durga",
    "hanuman", "ganesh", "ganesha", "saraswati", "lakshmi", "parvati",
    "temple", "mandir", "puja", "darshan", "prasad", "aarti", "bhajan",
    "hinduism", "buddha", "buddhism", "jain", "jainism", "sikhism",
    "sufi", "bhakti", "devotion", "scripture", "mythology", "mahakavya",
    "indus valley", "harappan", "mughal", "maurya", "gupta", "maratha",
    "ayurveda", "yoga", "meditation", "sanskrit", "ashram", "sadhu",
    "diwali", "holi", "navratri", "dussehra", "janmashtami", "pongal",
    "eid", "ramzan", "christmas", "guru nanak", "sikh", "gurudwara",
)


# â”€â”€ Dataclass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass(frozen=True)
class TextModerationResult:
    scores: dict[str, float]           # decision_engine-compatible scores
    text_scores: dict[str, float]      # detailed per-model breakdown
    detected_language: str             # ISO 639-1 from FastText, "unknown" on failure
    llama_result: dict | None          # raw Llama JSON output
    pipeline_error: bool = False
    error_reason: str | None = None


# â”€â”€ Lazy model singletons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_detoxify_model = None
_fasttext_model = None


def _get_detoxify():
    global _detoxify_model
    if _detoxify_model is not None:
        return _detoxify_model
    try:
        from detoxify import Detoxify
        try:
            _detoxify_model = Detoxify(DETOXIFY_MODEL)
            logger.info("Detoxify '%s' model loaded", DETOXIFY_MODEL)
        except Exception:
            logger.warning("Detoxify multilingual failed; falling back to 'unbiased'")
            _detoxify_model = Detoxify("unbiased")
            logger.info("Detoxify 'unbiased' model loaded")
    except ImportError:
        logger.warning("detoxify not installed â€” toxicity scoring disabled")
        _detoxify_model = None
    return _detoxify_model


def _get_fasttext():
    global _fasttext_model
    if _fasttext_model is not None:
        return _fasttext_model
    try:
        import fasttext
        import os
        if os.path.exists(FASTTEXT_MODEL_PATH):
            _fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            logger.info("FastText language model loaded from %s", FASTTEXT_MODEL_PATH)
        else:
            logger.warning(
                "FastText model not found at %s â€” language detection disabled",
                FASTTEXT_MODEL_PATH,
            )
    except ImportError:
        logger.warning("fasttext not installed â€” language detection disabled")
    return _fasttext_model


# â”€â”€ Internal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _normalize(text: str) -> str:
    return WHITESPACE.sub(" ", (text or "")).strip().lower()


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _phrase_count(text: str, phrases: tuple[str, ...]) -> int:
    return sum(1 for p in phrases if p in text)


def _phrase_score(text: str, phrases: tuple[str, ...], base: float = 0.35) -> float:
    n = _phrase_count(text, phrases)
    if n == 0:
        return 0.0
    return _clamp(base + (n - 1) * 0.15)


def _llama_to_risk(llama_result: dict | None) -> tuple[float, float]:
    """Return (llama_risk_score, llama_approves)."""
    if not llama_result:
        return 0.5, 0.0
    decision = str(llama_result.get("decision", "UNDER_REVIEW")).upper()
    confidence = _clamp(float(llama_result.get("confidence", 0.5)))
    if decision == "REJECTED":
        return confidence, 0.0
    if decision == "APPROVED":
        return 1.0 - confidence, confidence
    return 0.5, 0.0


# â”€â”€ Per-model scoring functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _detoxify_scores(text: str) -> dict[str, float]:
    """Run Detoxify and map its outputs to decision_engine score keys."""
    model = _get_detoxify()
    if model is None:
        return {}
    try:
        sanitized = text.replace("\x00", "").encode("utf-8", "replace").decode("utf-8")
        raw: dict = model.predict(sanitized[:_MAX_TEXT_LEN])
        # raw keys: toxicity, severe_toxicity, obscene, threat,
        #           insult, identity_attack, sexual_explicit
        adult = _clamp(max(
            float(raw.get("obscene", 0)) * 0.75,
            float(raw.get("sexual_explicit", 0)) * 0.90,
        ))
        violence = _clamp(float(raw.get("threat", 0)) * 0.90)
        harassment = _clamp(max(
            float(raw.get("insult", 0)),
            float(raw.get("severe_toxicity", 0)),
        ))
        hate_speech = _clamp(float(raw.get("identity_attack", 0)) * 1.10)
        return {
            "_detoxify_adult": adult,
            "_detoxify_violence": violence,
            "_detoxify_harassment": harassment,
            "_detoxify_hate_speech": hate_speech,
            "_detoxify_toxicity": _clamp(float(raw.get("toxicity", 0))),
        }
    except Exception:
        logger.exception("Detoxify inference failed")
        return {}


def _detect_language(text: str) -> str:
    """Return ISO 639-1 language code or 'unknown'."""
    model = _get_fasttext()
    if model is None:
        return "unknown"
    try:
        clean = text.replace("\n", " ")[:500]
        labels, _ = model.predict(clean, k=1)
        if labels:
            return labels[0].replace("__label__", "")
    except Exception:
        logger.warning("FastText language detection failed", exc_info=True)
    return "unknown"


def _text_rule_scores(text: str) -> dict[str, float]:
    """Reuse existing text_safety phrase rules."""
    from backend.pipeline.text_safety import analyze_text_safety
    return analyze_text_safety(ocr_text=text, caption=None)


def _promotion_scores(text: str) -> dict[str, float]:
    """Detect marketing, course, and political content."""
    marketing_count = float(_phrase_count(text, _MARKETING_PHRASES))
    course_score = _phrase_score(text, _COURSE_PHRASES, base=0.55)
    political_topic = _phrase_score(text, _POLITICAL_TOPIC_PHRASES, base=0.30)
    political_campaign = _phrase_score(text, _POLITICAL_PHRASES, base=0.45)

    # Overall promotion_score: marketing OR course is stronger
    promo = _clamp(max(
        min(marketing_count / 4.0, 1.0) * 0.85,
        course_score * 0.90,
    ))

    return {
        "marketing_keyword_count": marketing_count,
        "course_promotion_score": course_score,
        "political_score": political_topic,
        "political_campaign_score": political_campaign,
        "promotion_score": promo,
    }


def _violence_text_scores(text: str) -> dict[str, float]:
    """Detect animal cruelty and human killing language."""
    animal = _phrase_score(text, _ANIMAL_CRUELTY_PHRASES, base=0.55)
    killing = _phrase_score(text, _HUMAN_KILLING_PHRASES, base=0.50)
    return {
        "animal_cruelty_text_score": animal,
        "human_killing_text_score": killing,
    }


def _heritage_text_score(text: str) -> float:
    """Estimate heritage relevance from keyword presence."""
    count = sum(1 for kw in _HERITAGE_KEYWORDS if kw in text)
    if count == 0:
        return 0.0
    if count >= 4:
        return 0.70
    if count >= 2:
        return 0.55
    return 0.35


def _default_text_scores() -> dict[str, float]:
    return {
        # Standard image-pipeline keys (set to 0 for text â€” won't trigger image rules)
        "adult_score": 0.0,
        "heritage_score": 0.0,
        "content_quality_score": 0.0,
        "child_safety_score": 0.0,
        "child_presence_score": 0.0,
        "violence_self_harm_score": 0.0,
        "weapon_score": 0.0,
        "blood_score": 0.0,
        "self_harm_score": 0.0,
        "self_harm_text_score": 0.0,
        "promotion_score": 0.0,
        "advertising_score": 0.0,
        "affiliate_score": 0.0,
        "social_media_score": 0.0,
        "terrorism_score": 0.0,
        "drug_trafficking_score": 0.0,
        "human_trafficking_score": 0.0,
        "fraud_score": 0.0,
        "hate_speech_score": 0.0,
        "harassment_score": 0.0,
        "misinformation_score": 0.0,
        "pii_score": 0.0,
        "llama_risk_score": 0.0,
        "llama_approves": 0.0,
        "ensemble_risk_score": 0.0,
        "uncertainty_score": 0.0,
        # Text-specific Tier 0 keys
        "nsfw_score": 0.0,
        "visual_explicit_indicator": 0.0,
        "marketing_keyword_count": 0.0,
        "course_promotion_score": 0.0,
        "political_score": 0.0,
        "political_campaign_score": 0.0,
        "animal_cruelty_text_score": 0.0,
        "human_killing_text_score": 0.0,
        # Video keys (always 0 for text)
        "video_unsafe_frame_ratio": 0.0,
        "video_consecutive_unsafe_frames": 0.0,
    }


# â”€â”€ Score fusion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fuse_scores(
    detoxify: dict[str, float],
    text_rules: dict[str, float],
    promotion: dict[str, float],
    violence_text: dict[str, float],
    heritage: float,
    llama_result: dict | None,
) -> dict[str, float]:
    """Merge all signal sources into a decision_engine-compatible scores dict."""

    scores = _default_text_scores()

    # â”€â”€ Adult content â”€â”€
    scores["adult_score"] = _clamp(max(
        detoxify.get("_detoxify_adult", 0.0),
    ))
    scores["nsfw_score"] = scores["adult_score"]

    # â”€â”€ Violence / self-harm â”€â”€
    scores["violence_self_harm_score"] = _clamp(max(
        detoxify.get("_detoxify_violence", 0.0),
        text_rules.get("harassment_score", 0.0) * 0.60,
    ))
    scores["self_harm_text_score"] = text_rules.get("self_harm_text_score", 0.0)
    scores["self_harm_score"] = scores["self_harm_text_score"]

    # â”€â”€ Hate speech â”€â”€
    scores["hate_speech_score"] = _clamp(max(
        detoxify.get("_detoxify_hate_speech", 0.0),
        text_rules.get("hate_speech_score", 0.0),
    ))

    # â”€â”€ Harassment â”€â”€
    scores["harassment_score"] = _clamp(max(
        detoxify.get("_detoxify_harassment", 0.0),
        text_rules.get("harassment_score", 0.0),
    ))

    # â”€â”€ Terrorism / fraud / misinformation â”€â”€
    scores["terrorism_score"] = text_rules.get("terrorism_score", 0.0)
    scores["fraud_score"] = text_rules.get("fraud_score", 0.0)
    scores["misinformation_score"] = text_rules.get("misinformation_score", 0.0)

    # â”€â”€ Promotion â”€â”€
    scores["promotion_score"] = promotion.get("promotion_score", 0.0)
    scores["marketing_keyword_count"] = promotion.get("marketing_keyword_count", 0.0)
    scores["course_promotion_score"] = promotion.get("course_promotion_score", 0.0)
    scores["political_score"] = promotion.get("political_score", 0.0)
    scores["political_campaign_score"] = promotion.get("political_campaign_score", 0.0)

    # â”€â”€ Violence text signals â”€â”€
    scores["animal_cruelty_text_score"] = violence_text.get("animal_cruelty_text_score", 0.0)
    scores["human_killing_text_score"] = violence_text.get("human_killing_text_score", 0.0)

    # â”€â”€ Heritage (boosts ambiguous content toward UNDER_REVIEW) â”€â”€
    scores["heritage_score"] = heritage

    # â”€â”€ Llama â”€â”€
    llama_risk, llama_approves = _llama_to_risk(llama_result)
    scores["llama_risk_score"] = llama_risk
    scores["llama_approves"] = llama_approves

    # â”€â”€ Ensemble risk (max-fusion of primary signals) â”€â”€
    primary_risks = [
        scores["adult_score"],
        scores["violence_self_harm_score"],
        scores["hate_speech_score"],
        scores["harassment_score"],
        scores["terrorism_score"] * 1.10,
        scores["fraud_score"],
        scores["self_harm_text_score"],
        detoxify.get("_detoxify_toxicity", 0.0) * 0.80,
    ]
    scores["ensemble_risk_score"] = _clamp(
        max(primary_risks, default=0.0) * 0.90 + llama_risk * 0.10
    )

    return scores


# â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def moderate_text(
    text: str,
    metadata: dict | None = None,
) -> TextModerationResult:
    """Moderate a text post.

    Args:
        text: Raw text content of the post.
        metadata: Optional dict (currently unused; reserved for future context).

    Returns:
        TextModerationResult with a scores dict compatible with
        decision_engine.decide_with_reason_code().
    """
    if not text or not text.strip():
        return TextModerationResult(
            scores=_default_text_scores(),
            text_scores={},
            detected_language="unknown",
            llama_result=None,
            pipeline_error=True,
            error_reason="Empty text content.",
        )

    logger.info("Text moderation pipeline started (len=%d)", len(text))

    normalized = _normalize(text)

    try:
        # Stage 1 â€” Language detection
        language = _detect_language(text)
        logger.info("Detected language: %s", language)

        # Stage 2 â€” Rule-based text safety
        text_rules = _text_rule_scores(text)

        # Stage 3 â€” Detoxify ML scores
        detoxify = _detoxify_scores(normalized)

        # Stage 4 â€” Promotion / keyword detection
        promotion = _promotion_scores(normalized)

        # Stage 5 â€” Violence / cruelty text detection
        violence_text = _violence_text_scores(normalized)

        # Stage 6 â€” Heritage keyword scoring
        heritage = _heritage_text_score(normalized)

        # Stage 7 â€” Preliminary score fusion (for Llama context)
        pre_scores = _fuse_scores(
            detoxify, text_rules, promotion, violence_text, heritage, None
        )

        # Stage 8 â€” Llama reasoning
        from backend.pipeline.vlm_engine import reason_text_moderation
        llama_result = reason_text_moderation(
            text,
            adult_score=pre_scores["adult_score"],
            hate_speech_score=pre_scores["hate_speech_score"],
            harassment_score=pre_scores["harassment_score"],
            violence_score=pre_scores["violence_self_harm_score"],
            terrorism_score=pre_scores["terrorism_score"],
            fraud_score=pre_scores["fraud_score"],
            promotion_score=pre_scores["promotion_score"],
            self_harm_score=pre_scores["self_harm_text_score"],
            language=language,
        )

        # Stage 9 â€” Final score fusion with Llama result
        final_scores = _fuse_scores(
            detoxify, text_rules, promotion, violence_text, heritage, llama_result
        )

        text_scores_detail = {
            **detoxify,
            **text_rules,
            **promotion,
            **violence_text,
            "heritage_score": heritage,
            "detected_language": language,
        }

        logger.info(
            "Text moderation completed (ensemble=%.3f, decision=%s)",
            final_scores["ensemble_risk_score"],
            llama_result.get("decision") if llama_result else "n/a",
        )

        return TextModerationResult(
            scores=final_scores,
            text_scores=text_scores_detail,
            detected_language=language,
            llama_result=llama_result,
        )

    except Exception as exc:
        logger.exception("Text moderation pipeline failed")
        return TextModerationResult(
            scores=_default_text_scores(),
            text_scores={},
            detected_language="unknown",
            llama_result=None,
            pipeline_error=True,
            error_reason=str(exc),
        )

