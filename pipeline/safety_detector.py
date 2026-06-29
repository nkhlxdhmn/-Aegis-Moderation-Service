"""Violence and self-harm signal fusion for moderation.

Combines YOLO11x object detections with SigLIP2 safety prompt scores.
Key fix over the previous version: the agreement formula no longer amplifies
single-model noise. Agreement only contributes when BOTH models fire above
their individual noise floors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from typing import Any

logger = logging.getLogger(__name__)

WEAPON_LABELS = {
    "weapon", "gun", "rifle", "knife", "pistol", "firearm", "handgun", "sword",
}
BLOOD_LABELS = {
    "blood", "gore", "dead body", "corpse", "body",
}
VIOLENCE_LABELS = {
    "violence", "violent", "murder", "assault", "attack", "fight",
}
SELF_HARM_LABELS = {
    "self harm", "self-harm", "self_harm", "suicide", "suicidal",
}

SAFETY_PROMPTS = (
    "a person being physically attacked or assaulted",
    "a gun or firearm being used to threaten or shoot someone",
    "a knife or sharp weapon being used to harm a person",
    "graphic blood and gore from a real injury or violence",
    "a real dead body or human remains from a crime or accident",
    "explicit real-world violence or murder",
    "a person cutting themselves or engaging in self-harm",
    "content depicting a suicide attempt",
    "explicit pornographic sexual content between adults",
    "adult sexual nudity in an explicit sexual context",
)

# SigLIP2 sigmoid noise floor: ~0.01–0.08 for unrelated pairs.
# Scores above this threshold are treated as a genuine signal.
SIGLIP_SIGNAL_FLOOR = 0.10
# YOLO confidence floor for safety signal
YOLO_SIGNAL_FLOOR = 0.30


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", " ").replace("_", " ")


def _yolo_confidence(detection: Mapping[str, Any]) -> float:
    return _clamp(
        detection.get("confidence")
        or detection.get("score")
        or detection.get("probability")
        or 0.0
    )


def _yolo_label(detection: Mapping[str, Any]) -> str:
    return _norm(
        detection.get("class")
        or detection.get("label")
        or detection.get("name")
        or detection.get("class_name")
        or ""
    )


def _norm_set(labels: set[str]) -> set[str]:
    return {_norm(l) for l in labels}


def _max_yolo_buckets(
    yolo_detections: Sequence[Mapping[str, Any]],
) -> tuple[float, float, float, float]:
    weapon_terms = _norm_set(WEAPON_LABELS)
    blood_terms = _norm_set(BLOOD_LABELS)
    self_harm_terms = _norm_set(SELF_HARM_LABELS)
    violence_terms = _norm_set(VIOLENCE_LABELS)

    weapon = blood = self_harm = violence = 0.0
    for det in yolo_detections:
        if not isinstance(det, Mapping):
            continue
        label = _yolo_label(det)
        conf = _yolo_confidence(det)
        if label in weapon_terms:
            weapon = max(weapon, conf)
        elif label in blood_terms:
            blood = max(blood, conf)
        elif label in self_harm_terms:
            self_harm = max(self_harm, conf)
        elif label in violence_terms:
            violence = max(violence, conf)

    return weapon, blood, self_harm, violence


def _flatten_scores(scores: Any) -> dict[str, float]:
    """Flatten one level of nesting from SigLIP2 or legacy clip score dicts."""
    if not isinstance(scores, Mapping):
        return {}
    flat: dict[str, float] = {}
    for k, v in scores.items():
        if isinstance(v, Mapping):
            # nested dict — absorb its contents
            for inner_k, inner_v in v.items():
                try:
                    flat[str(inner_k)] = float(inner_v)
                except (TypeError, ValueError):
                    pass
        else:
            try:
                flat[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
    return flat


def _max_clip_buckets(
    clip_safety_scores: Any,
) -> tuple[float, float, float, float]:
    """Extract bucket scores from SigLIP2 safety scores dict."""
    flat = _flatten_scores(clip_safety_scores)
    if not flat:
        return 0.0, 0.0, 0.0, 0.0

    weapon_terms = _norm_set(WEAPON_LABELS) | {"gun", "knife", "firearm", "rifle"}
    blood_terms = _norm_set(BLOOD_LABELS) | {"gore"}
    self_harm_terms = _norm_set(SELF_HARM_LABELS) | {"suicide"}
    violence_terms = _norm_set(VIOLENCE_LABELS) | {"murder", "assault"}

    weapon = blood = self_harm = violence = 0.0
    for raw_prompt, score in flat.items():
        p = _norm(raw_prompt)
        s = _clamp(score)
        # Match to bucket by keyword presence in the contextualized prompt
        if any(t in p for t in weapon_terms):
            weapon = max(weapon, s)
        if any(t in p for t in blood_terms):
            blood = max(blood, s)
        if any(t in p for t in self_harm_terms):
            self_harm = max(self_harm, s)
        if any(t in p for t in violence_terms):
            violence = max(violence, s)

    return weapon, blood, self_harm, violence


def _fuse_pair(yolo_score: float, clip_score: float) -> float:
    """Fuse one YOLO score and one SigLIP2 score for a safety bucket.

    agreement fires only when both signals exceed the 0.15 noise floor,
    preventing SigLIP noise from amplifying a zero YOLO score.
    max-based formula avoids double-counting (no additive sum).
    """
    agreement = (yolo_score * clip_score) ** 0.5 if yolo_score > 0.15 and clip_score > 0.15 else 0.0
    return _clamp(max(yolo_score * 0.85, clip_score * 0.60, agreement * 0.95))


def analyze_safety(
    yolo_detections: Sequence[Mapping[str, Any]] | None,
    clip_safety_scores: Any,
) -> dict[str, float]:
    """Combine YOLO11x and SigLIP2 violence/weapon/self-harm signals."""
    logger.info("Violence safety signal fusion started")
    detections = yolo_detections or []

    yolo_weapon, yolo_blood, yolo_self_harm, yolo_violence = _max_yolo_buckets(detections)
    clip_weapon, clip_blood, clip_self_harm, clip_violence = _max_clip_buckets(clip_safety_scores)

    weapon_score = _fuse_pair(yolo_weapon, clip_weapon)
    blood_score = _fuse_pair(yolo_blood, clip_blood)
    self_harm_score = _fuse_pair(yolo_self_harm, clip_self_harm)
    direct_violence_score = _fuse_pair(yolo_violence, clip_violence)
    violence_self_harm_score = _clamp(
        max(weapon_score, blood_score, self_harm_score, direct_violence_score)
    )

    logger.info("Violence safety signal fusion completed")
    return {
        "weapon_score": weapon_score,
        "blood_score": blood_score,
        "self_harm_score": self_harm_score,
        "violence_self_harm_score": violence_self_harm_score,
    }


# Compatibility alias
def get_safety_scores(
    yolo_detections: Sequence[Mapping[str, Any]] | None,
    clip_safety_scores: Any,
) -> dict[str, float]:
    return analyze_safety(yolo_detections, clip_safety_scores)
