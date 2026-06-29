"""Child-presence signal fusion for moderation.

Combines YOLO11x object detections with SigLIP2 child-safety prompt scores.

Key fixes over the previous version:
  1. Agreement only fires when BOTH YOLO and SigLIP detect child-related signals
     above noise floor. When YOLO fires zero, agreement = SigLIP alone; the
     previous formula multiplied that by 0.70 (> 0.60), silently boosting noise.
  2. person_supported_clip_score weight reduced: detecting "person" (which YOLO
     fires on every carved temple figure) no longer dominates the child score.
  3. SigLIP2 contextualized prompts have a much lower noise floor (~0.01–0.08),
     so a raw score > 0.10 is already a genuine signal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from typing import Any

logger = logging.getLogger(__name__)

CHILD_LABELS = {
    "child", "infant", "toddler", "minor", "school child", "school_child",
}
CHILD_ALIASES = {
    "baby", "kid", "kids", "young child", "young_child",
    "small child", "small_child", "children",
}
PERSON_LABELS = {"person", "people", "human"}

CHILD_PROMPTS = (
    "child sexual exploitation or abuse material",
    "a child in a dangerous, harmful, or threatening situation",
    "inappropriate sexual content involving a minor",
    "a child being abused or harmed by an adult",
)

# SigLIP2 sigmoid noise floor for unrelated pairs
SIGLIP_SIGNAL_FLOOR = 0.10
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


def _max_yolo_scores(
    yolo_detections: Sequence[Mapping[str, Any]],
) -> tuple[float, float]:
    child_terms = {_norm(l) for l in CHILD_LABELS | CHILD_ALIASES}
    person_terms = {_norm(l) for l in PERSON_LABELS}

    child_score = person_score = 0.0
    for det in yolo_detections:
        if not isinstance(det, Mapping):
            continue
        label = _yolo_label(det)
        conf = _yolo_confidence(det)
        if label in child_terms:
            child_score = max(child_score, conf)
        elif label in person_terms:
            person_score = max(person_score, conf)

    return child_score, person_score


CHILD_KEY_TERMS = {
    "child", "infant", "toddler", "minor", "baby", "kid", "kids", "children",
    "underage", "juvenile", "abuse", "exploit", "school",
}


def _flatten_child_scores(scores: Any) -> dict[str, float]:
    """Flatten one level of nesting and return only child-related entries."""
    if not isinstance(scores, Mapping):
        return {}
    flat: dict[str, float] = {}
    for k, v in scores.items():
        if isinstance(v, Mapping):
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

    # Keep only keys that contain a child-related term
    return {
        k: v
        for k, v in flat.items()
        if any(term in _norm(k) for term in CHILD_KEY_TERMS)
    }


def _max_clip_child_score(clip_child_scores: Any) -> float:
    """Return highest SigLIP2 score across child-safety prompts.

    Only scores whose key contains a child-related term are considered.
    Keys like "adult" or "vehicle" are filtered out even if their values
    are high, preventing false-positive elevation from unrelated scores.
    """
    child_entries = _flatten_child_scores(clip_child_scores)
    if not child_entries:
        return 0.0
    return max((_clamp(v) for v in child_entries.values()), default=0.0)


def analyze_child_safety(
    yolo_detections: Sequence[Mapping[str, Any]] | None,
    clip_child_scores: Any,
) -> dict[str, float]:
    """Combine YOLO11x and SigLIP2 signals into child-presence safety scores."""
    logger.info("Child safety signal fusion started")
    detections = yolo_detections or []

    yolo_child_score, yolo_person_score = _max_yolo_scores(detections)
    clip_child_score = _max_clip_child_score(clip_child_scores)

    # ── child_presence_score ──────────────────────────────────────────────────
    # "Are there children visible?" — broader signal including person detections
    # Person detections get a low weight (0.15) because temple carvings/statues
    # are routinely classified as "person" by general-purpose YOLO models.
    weak_person_presence = yolo_person_score * 0.25
    child_presence_score = _clamp(
        max(yolo_child_score, clip_child_score, weak_person_presence)
    )

    # ── child_safety_score ────────────────────────────────────────────────────
    # Spec formula: max(a*0.85, b*0.60, agreement*0.95)
    # where a = yolo_child_score, b = clip_child_score
    # agreement fires only when both exceed 0.15 noise floor
    agreement = (
        (yolo_child_score * clip_child_score) ** 0.5
        if yolo_child_score > 0.15 and clip_child_score > 0.15
        else 0.0
    )
    base = max(yolo_child_score * 0.85, clip_child_score * 0.60, agreement * 0.95)

    # Person-supported clip: real person + SigLIP child-safety signal.
    # Very low weight (0.15) so temple carvings don't cause false positives.
    person_boost = 0.0
    if yolo_person_score > 0.15 and clip_child_score > 0.15:
        person_boost = min(yolo_person_score, clip_child_score) * 0.15

    child_safety_score = _clamp(max(base, person_boost))

    logger.info("Child safety signal fusion completed")
    return {
        "child_presence_score": child_presence_score,
        "child_safety_score": child_safety_score,
    }


# Compatibility alias
def get_child_safety_scores(
    yolo_detections: Sequence[Mapping[str, Any]] | None,
    clip_child_scores: Any,
) -> dict[str, float]:
    return analyze_child_safety(yolo_detections, clip_child_scores)
