"""Moderation report normalization and rule-based decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CATEGORY_LABELS: dict[str, str] = {
    "adult_content": "Adult Content",
    "violence": "Violence",
    "graphic_violence": "Graphic Violence",
    "weapons": "Weapons",
    "drugs": "Drugs",
    "alcohol": "Alcohol",
    "smoking": "Smoking",
    "gambling": "Gambling",
    "blood": "Blood",
    "nudity": "Nudity",
    "suggestive_content": "Suggestive Content",
    "hate_symbol": "Hate Symbol",
    "hate_speech": "Hate Speech",
    "toxic_text": "Toxic Text",
    "abusive_text": "Abusive Text",
    "political_propaganda": "Political Propaganda",
    "religious_extremism": "Religious Extremism",
    "child_safety_risk": "Child Safety Risk",
    "pii_detection": "PII Detection",
    "qr_code": "QR Code",
    "document": "Document",
    "watermark": "Watermark",
    "spam": "Spam",
    "scam": "Scam",
    "misinformation": "Misinformation",
    "self_harm": "Self Harm",
    "medical_content": "Medical Content",
}

SCORE_ALIASES: dict[str, tuple[str, ...]] = {
    "adult_content": ("adult_score", "nsfw_score", "explicit_score"),
    "violence": ("violence_score", "violence_self_harm_score"),
    "graphic_violence": ("graphic_violence_score", "gore_score", "blood_score"),
    "weapons": ("weapon_score", "weapons_score"),
    "drugs": ("drug_score", "drugs_score", "drug_trafficking_score"),
    "alcohol": ("alcohol_score",),
    "smoking": ("smoking_score",),
    "gambling": ("gambling_score",),
    "blood": ("blood_score",),
    "nudity": ("nudity_score", "adult_score"),
    "suggestive_content": ("suggestive_score", "racy_score"),
    "hate_symbol": ("hate_symbol_score",),
    "hate_speech": ("hate_speech_score", "ml_hate_score"),
    "toxic_text": ("toxic_text_score", "ml_toxicity_score"),
    "abusive_text": ("abusive_text_score", "harassment_score", "text_classifier_score"),
    "political_propaganda": (
        "political_propaganda_score",
        "political_campaign_score",
        "political_score",
    ),
    "religious_extremism": ("religious_extremism_score", "extremism_score"),
    "child_safety_risk": ("child_safety_score", "child_safety_risk_score"),
    "pii_detection": ("pii_score", "pii_detection_score"),
    "qr_code": ("qr_code_score",),
    "document": ("document_score",),
    "watermark": ("watermark_score",),
    "spam": ("spam_score", "promotion_score", "advertising_score"),
    "scam": ("scam_score", "fraud_score"),
    "misinformation": ("misinformation_score",),
    "self_harm": ("self_harm_score", "self_harm_text_score", "violence_self_harm_score"),
    "medical_content": ("medical_content_score", "medical_score"),
}


@dataclass(frozen=True)
class Decision:
    """Human-readable moderation decision derived from category scores."""

    risk_level: str
    decision: str
    recommendation: str


def _as_percent(value: Any) -> float:
    """Convert model output into a clamped percentage."""

    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score <= 1.0:
        score *= 100.0
    return round(max(0.0, min(score, 100.0)), 1)


def _max_score(raw_scores: Mapping[str, Any], aliases: tuple[str, ...]) -> float:
    return max((_as_percent(raw_scores.get(alias, 0.0)) for alias in aliases), default=0.0)


def normalize_categories(raw_scores: Mapping[str, Any]) -> dict[str, float]:
    """Return the public category taxonomy with percentage scores."""

    categories = {key: _max_score(raw_scores, aliases) for key, aliases in SCORE_ALIASES.items()}
    categories["graphic_violence"] = max(
        categories["graphic_violence"], categories["violence"] * 0.8
    )
    categories["nudity"] = max(categories["nudity"], categories["adult_content"] * 0.6)
    return {key: round(value, 1) for key, value in categories.items()}


def calculate_overall_score(
    categories: Mapping[str, float], raw_scores: Mapping[str, Any]
) -> float:
    """Calculate the headline risk score from category and ensemble signals."""

    ensemble = _as_percent(raw_scores.get("ensemble_risk_score", 0.0))
    return round(max([ensemble, *categories.values()]), 1)


def decide(overall_score: float, categories: Mapping[str, float]) -> Decision:
    """Apply transparent moderation rules to the normalized report."""

    if (
        overall_score >= 90
        or categories.get("adult_content", 0) >= 90
        or categories.get("child_safety_risk", 0) >= 80
    ):
        return Decision("CRITICAL", "Reject", "Reject")
    if overall_score >= 70:
        return Decision("HIGH RISK", "Review Required", "Human Review")
    if overall_score >= 40 or categories.get("weapons", 0) >= 70:
        return Decision("MEDIUM RISK", "Review Required", "Human Review")
    if overall_score >= 20:
        return Decision("LOW RISK", "Accept", "Allow with Monitoring")
    return Decision("SAFE", "Accept", "Allow")


def _object_name(item: Any) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        for key in ("label", "class", "name", "object"):
            value = item.get(key)
            if value:
                return str(value)
    return None


def extract_objects(raw_objects: Any) -> list[str]:
    """Return deduplicated object labels from detector output."""

    if not isinstance(raw_objects, list):
        return []
    seen: set[str] = set()
    objects: list[str] = []
    for item in raw_objects:
        name = _object_name(item)
        if not name:
            continue
        normalized = name.strip()
        key = normalized.lower()
        if key and key not in seen:
            seen.add(key)
            objects.append(normalized)
    return objects


def build_report(pipeline_result: Any) -> dict[str, Any]:
    """Build the public standalone moderation report from pipeline output."""

    raw_scores: Mapping[str, Any] = getattr(pipeline_result, "scores", {}) or {}
    category_scores: Mapping[str, Any] = getattr(pipeline_result, "category_scores", {}) or {}
    merged_scores = {**raw_scores, **category_scores}
    categories = normalize_categories(merged_scores)
    overall_score = calculate_overall_score(categories, merged_scores)
    decision = decide(overall_score, categories)
    pipeline_error = bool(getattr(pipeline_result, "pipeline_error", False))

    if pipeline_error and overall_score < 40:
        overall_score = 40.0
        decision = Decision("MEDIUM RISK", "Review Required", "Human Review")

    return {
        "overall_score": overall_score,
        "risk_level": decision.risk_level,
        "decision": decision.decision,
        "categories": categories,
        "category_labels": CATEGORY_LABELS,
        "objects": extract_objects(getattr(pipeline_result, "detected_objects", [])),
        "ocr_text": getattr(pipeline_result, "ocr_text", "") or "",
        "recommendation": decision.recommendation,
        "caption": getattr(pipeline_result, "generated_caption", "") or "",
        "image_hash": getattr(pipeline_result, "image_hash", None),
        "model_versions": getattr(pipeline_result, "model_versions", {}) or {},
        "error": getattr(pipeline_result, "error_reason", None) if pipeline_error else None,
    }
