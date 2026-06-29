"""Vision-Language Model engine — BLIP captioning + rule-based moderation.

Models:
  Salesforce/blip-image-captioning-large  — generates 3 diverse captions.

Llama and Qwen have been removed. Moderation decisions use the rule-based
decision engine in pipeline/decision_engine.py which covers all safety tiers
without requiring a generative LLM.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
BLIP2_MODEL_ID = BLIP_MODEL_ID  # backward-compat alias
LLAMA_MODEL_ID = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"  # kept for compat
DEVICE = os.getenv("VLM_DEVICE", "cuda:0")


class ModelInferenceError(RuntimeError):
    """Raised when a VLM model cannot process input."""


# ── BLIP state ────────────────────────────────────────────────────────────────


@dataclass
class _BLIPState:
    model: Any
    processor: Any
    torch: Any
    device: str


_blip_state: _BLIPState | None = None
_blip2_state = None  # backward-compat alias — always mirrors _blip_state
_blip_lock = threading.Lock()


def _get_blip() -> _BLIPState:
    global _blip_state
    if _blip_state is not None:
        return _blip_state

    with _blip_lock:
        if _blip_state is not None:
            return _blip_state

        logger.info("Loading BLIP image-captioning-large on %s", DEVICE)
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor

            processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
            model = BlipForConditionalGeneration.from_pretrained(
                BLIP_MODEL_ID,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(DEVICE)
            model.eval()

            local = _BLIPState(model=model, processor=processor, torch=torch, device=DEVICE)
            _blip_state = local
        except Exception as exc:
            logger.exception("Failed to load BLIP")
            raise ModelInferenceError("BLIP failed to load") from exc

        logger.info("BLIP image-captioning-large loaded on %s", DEVICE)
    return _blip_state  # type: ignore[return-value]


def generate_caption(image_path: str) -> str:
    """Generate a single greedy BLIP caption.  Kept for backward compatibility."""
    captions = generate_captions(image_path, n=1)
    return captions[0] if captions else ""


def generate_captions(image_path: str, n: int = 3) -> list[str]:
    """Generate n diverse BLIP captions using different decoding strategies.

    Strategy:
      1. Greedy decode  — deterministic, most-likely caption.
      2. Beam search    — quality-optimised alternative (n ≥ 2).
      3. Temperature=1.2 sampling — diverse / edge-case coverage (n ≥ 3).

    Deduplicates identical captions.  Returns [] on any failure so the
    pipeline degrades gracefully.
    """
    logger.info("BLIP multi-caption generation started (n=%d)", n)
    try:
        state = _get_blip()
        torch = state.torch

        from PIL import Image

        with Image.open(image_path) as img:
            inputs = state.processor(
                images=img.convert("RGB"),
                return_tensors="pt",
            ).to(DEVICE, dtype=torch.float16)

        captions: list[str] = []
        with torch.inference_mode():
            # 1. Greedy (deterministic)
            ids = state.model.generate(**inputs, max_new_tokens=40, do_sample=False)
            cap = state.processor.decode(ids[0], skip_special_tokens=True).strip()
            if cap:
                captions.append(cap)

            if n >= 2:
                # 2. Beam search (quality)
                ids = state.model.generate(
                    **inputs,
                    max_new_tokens=40,
                    num_beams=3,
                    do_sample=False,
                )
                cap = state.processor.decode(ids[0], skip_special_tokens=True).strip()
                if cap and cap not in captions:
                    captions.append(cap)

            if n >= 3:
                # 3. Sampling with higher temperature (diversity)
                ids = state.model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=1.2,
                )
                cap = state.processor.decode(ids[0], skip_special_tokens=True).strip()
                if cap and cap not in captions:
                    captions.append(cap)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception:
        logger.exception("BLIP multi-caption generation failed; returning empty list")
        return []

    logger.info("BLIP captions generated: %s", captions)
    return captions


# ── Rule-based moderation stubs (replace Llama) ───────────────────────────────
# The decision_engine.py tiered rule cascade handles all safety decisions.
# These functions return neutral confidence so no Llama tier fires and the
# full rule cascade in decision_engine._evaluate() takes over.


def reason_moderation(
    nsfw_score: float,
    objects_detected: list[str],
    ocr_text: str,
    caption: str,
    blip_caption: str = "",
    blip_captions: list[str] | None = None,
    heritage_score: float = 0.0,
    child_safety_score: float = 0.0,
    violence_score: float = 0.0,
    weapon_score: float = 0.0,
    blip2_caption: str | None = None,
    qwen_description: str = "",
    qwen_verification: str = "",
) -> dict:
    """Return neutral confidence so decision_engine rule cascade takes over."""
    return {
        "decision": "UNDER_REVIEW",
        "reason": "Rule-based moderation active.",
        "confidence": 0.5,
        "category": "Uncategorized",
    }


def reason_text_moderation(
    text: str,
    *,
    adult_score: float = 0.0,
    hate_speech_score: float = 0.0,
    harassment_score: float = 0.0,
    violence_score: float = 0.0,
    terrorism_score: float = 0.0,
    fraud_score: float = 0.0,
    promotion_score: float = 0.0,
    self_harm_score: float = 0.0,
    language: str = "unknown",
) -> dict:
    """Return neutral confidence so decision_engine rule cascade takes over."""
    return {
        "decision": "UNDER_REVIEW",
        "reason": "Rule-based text moderation active.",
        "confidence": 0.5,
        "category": "Uncategorized",
    }
