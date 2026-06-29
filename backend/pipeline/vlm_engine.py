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
import json
import re
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

BLIP_MODEL_ID = "Salesforce/blip-image-captioning-large"
BLIP2_MODEL_ID = BLIP_MODEL_ID  # backward-compat alias
LLAMA_MODEL_ID = os.getenv(
    "LLAMA_MODEL_ID",
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
)
DEVICE = os.getenv("VLM_DEVICE", "cuda:0")
LLAMA_MAX_NEW_TOKENS = int(os.getenv("LLAMA_MAX_NEW_TOKENS", "192"))


class ModelInferenceError(RuntimeError):
    """Raised when a VLM model cannot process input."""


# ── BLIP state ────────────────────────────────────────────────────────────────


@dataclass
class _BLIPState:
    model: Any
    processor: Any
    torch: Any
    device: str
    dtype: Any


_blip_state: _BLIPState | None = None
_blip2_state = None  # backward-compat alias — always mirrors _blip_state
_blip_lock = threading.Lock()


@dataclass
class _LlamaState:
    model: Any
    tokenizer: Any
    torch: Any
    device: str


_llama_state: _LlamaState | None = None
_llama_lock = threading.Lock()


def _resolve_device(torch: Any) -> str:
    requested = DEVICE
    if not str(requested).startswith("cuda"):
        return requested
    if not torch.cuda.is_available():
        logger.warning("VLM_DEVICE=%s requested but CUDA is unavailable; using CPU", requested)
        return "cpu"
    try:
        index = int(str(requested).split(":", 1)[1]) if ":" in str(requested) else 0
    except ValueError:
        index = 0
    if index >= torch.cuda.device_count():
        logger.warning(
            "VLM_DEVICE=%s requested but only %d CUDA device(s) are visible; using cuda:0",
            requested,
            torch.cuda.device_count(),
        )
        return "cuda:0"
    return requested


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

            device = _resolve_device(torch)
            dtype = torch.float16 if device != "cpu" else torch.float32
            processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
            model = BlipForConditionalGeneration.from_pretrained(
                BLIP_MODEL_ID,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            model.eval()

            local = _BLIPState(
                model=model,
                processor=processor,
                torch=torch,
                device=device,
                dtype=dtype,
            )
            _blip_state = local
        except Exception as exc:
            logger.exception("Failed to load BLIP")
            raise ModelInferenceError("BLIP failed to load") from exc

        logger.info("BLIP image-captioning-large loaded on %s", _blip_state.device)
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
            ).to(state.device, dtype=state.dtype)

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


def _get_llama() -> _LlamaState:
    global _llama_state
    if _llama_state is not None:
        return _llama_state

    with _llama_lock:
        if _llama_state is not None:
            return _llama_state

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = _resolve_device(torch)
            if device == "cpu":
                logger.warning("Loading Llama on CPU; this is slow and memory-heavy")
            dtype = torch.float16 if device != "cpu" else torch.float32
            logger.info("Loading Llama reasoning model %s on %s", LLAMA_MODEL_ID, device)

            tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID, trust_remote_code=True)
            model_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            if device != "cpu":
                model_kwargs["device_map"] = {"": device}

            model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_ID, **model_kwargs)
            if device == "cpu":
                model = model.to(device)
            model.eval()

            _llama_state = _LlamaState(
                model=model,
                tokenizer=tokenizer,
                torch=torch,
                device=device,
            )
        except Exception as exc:
            logger.exception("Failed to load Llama")
            raise ModelInferenceError("Llama failed to load") from exc

        logger.info("Llama reasoning model loaded on %s", _llama_state.device)
    return _llama_state


def _extract_json(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Llama response")
    data = json.loads(match.group(0))
    decision = str(data.get("decision", "UNDER_REVIEW")).upper()
    if decision not in {"APPROVED", "REJECTED", "UNDER_REVIEW"}:
        decision = "UNDER_REVIEW"
    confidence = float(data.get("confidence", 0.5))
    return {
        "decision": decision,
        "reason": str(data.get("reason", "Llama reasoning completed."))[:500],
        "confidence": max(0.0, min(1.0, confidence)),
        "category": str(data.get("category", "Uncategorized"))[:120],
    }


def _generate_llama_json(messages: list[dict[str, str]]) -> dict[str, Any]:
    state = _get_llama()
    tokenizer = state.tokenizer
    torch = state.torch

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(state.device)

    with torch.inference_mode():
        output_ids = state.model.generate(
            **inputs,
            max_new_tokens=LLAMA_MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _extract_json(text)


def _fallback_result() -> dict[str, Any]:
    return {
        "decision": "UNDER_REVIEW",
        "reason": "Llama reasoning unavailable.",
        "confidence": 0.5,
        "category": "Uncategorized",
    }


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
    """Use Llama to reason over fused image evidence and return a strict JSON verdict."""
    captions = blip_captions or []
    if blip_caption:
        captions.append(blip_caption)
    if blip2_caption:
        captions.append(blip2_caption)

    evidence = {
        "nsfw_score": round(float(nsfw_score), 4),
        "objects_detected": objects_detected,
        "ocr_text": (ocr_text or "")[:1200],
        "user_caption": (caption or "")[:500],
        "blip_captions": captions[:5],
        "heritage_score": round(float(heritage_score), 4),
        "child_safety_score": round(float(child_safety_score), 4),
        "violence_score": round(float(violence_score), 4),
        "weapon_score": round(float(weapon_score), 4),
        "qwen_description": (qwen_description or "")[:800],
        "qwen_verification": (qwen_verification or "")[:800],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a content moderation reasoning model. Return only JSON with "
                "keys decision, reason, confidence, category. decision must be one of "
                "APPROVED, REJECTED, UNDER_REVIEW. Be conservative for sexual, child "
                "safety, violence, self-harm, fraud, and terrorism signals. Protect "
                "benign heritage, religious, educational, and documentary content."
            ),
        },
        {
            "role": "user",
            "content": f"Moderate this image evidence:\n{json.dumps(evidence, ensure_ascii=False)}",
        },
    ]
    try:
        return _generate_llama_json(messages)
    except Exception:
        logger.exception("Llama image moderation failed")
        return _fallback_result()


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
    """Use Llama to reason over text moderation evidence and return a strict JSON verdict."""
    evidence = {
        "text": (text or "")[:2000],
        "adult_score": round(float(adult_score), 4),
        "hate_speech_score": round(float(hate_speech_score), 4),
        "harassment_score": round(float(harassment_score), 4),
        "violence_score": round(float(violence_score), 4),
        "terrorism_score": round(float(terrorism_score), 4),
        "fraud_score": round(float(fraud_score), 4),
        "promotion_score": round(float(promotion_score), 4),
        "self_harm_score": round(float(self_harm_score), 4),
        "language": language,
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a content moderation reasoning model. Return only JSON with "
                "keys decision, reason, confidence, category. decision must be one of "
                "APPROVED, REJECTED, UNDER_REVIEW. Be conservative for abuse, hate, "
                "harassment, sexual content, violence, self-harm, scams, and terrorism."
            ),
        },
        {
            "role": "user",
            "content": f"Moderate this text evidence:\n{json.dumps(evidence, ensure_ascii=False)}",
        },
    ]
    try:
        return _generate_llama_json(messages)
    except Exception:
        logger.exception("Llama text moderation failed")
        return _fallback_result()
