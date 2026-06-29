"""OpenNSFW2 ViT-L adult-content detection for the Aegis moderation pipeline.

Model: Falconsai/nsfw_image_detection (ViT-Large fine-tuned for NSFW detection).
This is the ViT-L equivalent of the Yahoo Open NSFW2 architecture, loaded via
the Hugging Face transformers library for GPU inference on cuda:0.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

NSFW_MODEL_ID = "Falconsai/nsfw_image_detection"
DEVICE = "cuda:0"

_state: dict[str, Any] | None = None
_state_lock = threading.Lock()


class ModelInferenceError(RuntimeError):
    """Raised when the NSFW model cannot process an image."""


def _get_state() -> dict[str, Any]:
    global _state
    if _state is not None:
        return _state

    with _state_lock:
        if _state is not None:
            return _state

        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForImageClassification

            from core.runtime import get_runtime

            device = get_runtime().torch_device
            dtype = torch.float16 if device != "cpu" else torch.float32
            logger.info("Loading OpenNSFW2 ViT-L on %s", device)

            processor = AutoFeatureExtractor.from_pretrained(
                NSFW_MODEL_ID,
                use_fast=True,
            )
            model = AutoModelForImageClassification.from_pretrained(
                NSFW_MODEL_ID,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            model.eval()
            torch.backends.cudnn.benchmark = True

            # Build label â†’ index map once
            id2label: dict[int, str] = getattr(model.config, "id2label", {0: "normal", 1: "nsfw"})
            nsfw_idx = next(
                (k for k, v in id2label.items() if "nsfw" in str(v).lower()),
                1,
            )

            local: dict[str, Any] = {
                "model": model,
                "processor": processor,
                "torch": torch,
                "nsfw_idx": nsfw_idx,
                "device": device,
                "dtype": dtype,
            }
            _state = local  # publish only after full init (double-checked locking)
        except Exception as exc:
            logger.exception("Failed to load OpenNSFW2 ViT-L")
            raise ModelInferenceError("OpenNSFW2 ViT-L failed to load") from exc

        logger.info("OpenNSFW2 ViT-L loaded on %s", device)
    return _state


def get_adult_score(image_path: str) -> float:
    """Return an NSFW probability score in [0, 1] for an image.

    Score interpretation:
        < 0.30  safe content
        0.30â€“0.50  review threshold (mildly suggestive)
        > 0.50  rejected as adult content
    """
    logger.info("OpenNSFW2 inference started")
    try:
        state = _get_state()
        torch = state["torch"]
        model = state["model"]
        processor = state["processor"]
        nsfw_idx = state["nsfw_idx"]

        from PIL import Image

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        device = state["device"]
        dtype = state["dtype"]

        inputs = processor(images=image, return_tensors="pt")
        # Cast pixel_values to match model dtype
        inputs = {
            k: v.to(device, dtype=dtype) if v.dtype == torch.float32 else v.to(device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():
            logits = model(**inputs).logits  # [1, num_classes]
            probs = torch.softmax(logits, dim=-1)  # calibrated class probabilities
            nsfw_score = float(probs[0, nsfw_idx].cpu())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.exception("OpenNSFW2 inference failed")
        if isinstance(exc, ModelInferenceError):
            raise
        raise ModelInferenceError("OpenNSFW2 inference failed") from exc

    logger.info("OpenNSFW2 inference completed: score=%.3f", nsfw_score)
    return max(0.0, min(1.0, nsfw_score))
