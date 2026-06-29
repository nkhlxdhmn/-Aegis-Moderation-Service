"""qwen_vl.py — Qwen2.5-VL removed; stubs return empty immediately.

Qwen2.5-VL-7B-Instruct was removed to reduce resource requirements.
BLIP captions (pipeline/vlm_engine.py) provide visual context instead.
All public functions are preserved for import compatibility.
"""

import logging

logger = logging.getLogger(__name__)

MODEL_ID: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # kept for compat


def describe_image(image_path: str) -> dict:
    """Stub — returns empty result without loading any model."""
    return {"description": "", "confidence": 0.5}


def verify_borderline(image_path: str, risk_score: float) -> dict:
    """Stub — returns empty result without loading any model."""
    return {"verification_reason": "", "confidence": 0.5}
