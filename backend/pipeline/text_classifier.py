"""Optional text abuse classifier hook.

Loads a fine-tuned HuggingFace sequence-classification model from a
configurable path.  When model weights are absent the classifier is
disabled and classify_text() returns a neutral result Ã¢â‚¬â€ the pipeline
continues without interruption and without any error.

Configuration
Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
  TEXT_CLASSIFIER_MODEL_DIR  (env var)
    Path to the model directory.
    Default: <service_root>/models/muril_abuse_final

Expected model layout
Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
  models/muril_abuse_final/
  Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ model.safetensors  OR  pytorch_model.bin
  Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ config.json
  Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ tokenizer.json
  Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ tokenizer_config.json
  Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ special_tokens_map.json

Labels
Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
  0 Ã¢â€ â€™ non_abusive
  1 Ã¢â€ â€™ abusive

Expected base model: google/muril-base-cased (236M-param multilingual BERT)
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SERVICE_ROOT = Path(__file__).resolve().parent.parent

_REQUIRED_CONFIG_FILES: frozenset[str] = frozenset(
    {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
)

# Neutral result returned when the classifier is disabled or inference fails.
_DISABLED_RESULT: dict[str, Any] = {
    "label": "non_abusive",
    "abuse_score": 0.0,
    "disabled": True,
}

_classifier_pipeline: Any = None
_classifier_disabled: bool = False  # set True once weights are confirmed absent
_classifier_lock = threading.Lock()


# Ã¢â€â‚¬Ã¢â€â‚¬ Configuration Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def _get_model_dir() -> Path:
    override = os.getenv("TEXT_CLASSIFIER_MODEL_DIR")
    return Path(override) if override else _SERVICE_ROOT / "models" / "muril_abuse_final"


def _weights_present(model_dir: Path) -> bool:
    """Return True if the directory has required config files and at least one weight file."""
    if not model_dir.is_dir():
        return False
    existing = {f.name for f in model_dir.iterdir() if f.is_file()}
    if _REQUIRED_CONFIG_FILES - existing:
        return False
    return "model.safetensors" in existing or "pytorch_model.bin" in existing


# Ã¢â€â‚¬Ã¢â€â‚¬ Singleton loader Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def load_text_classifier() -> None:
    """Attempt to load the text classifier.  No-op when weights are absent."""
    global _classifier_pipeline, _classifier_disabled

    if _classifier_disabled or _classifier_pipeline is not None:
        return

    with _classifier_lock:
        if _classifier_disabled or _classifier_pipeline is not None:
            return

        model_dir = _get_model_dir()
        if not _weights_present(model_dir):
            logger.info(
                "Text classifier disabled Ã¢â‚¬â€ weights not found at %s "
                "(will auto-enable when models/muril_abuse_final/ is populated)",
                model_dir,
            )
            _classifier_disabled = True
            return

        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = 0 if torch.cuda.is_available() else -1
            _classifier_pipeline = hf_pipeline(
                task="text-classification",
                model=str(model_dir),
                tokenizer=str(model_dir),
                device=device,
                truncation=True,
                max_length=128,
            )
            logger.info(
                "Text classifier loaded from %s (device=%s)",
                model_dir,
                "cuda:0" if device == 0 else "cpu",
            )
        except Exception:
            logger.exception("Text classifier load failed Ã¢â‚¬â€ disabling for this session")
            _classifier_disabled = True
            _classifier_pipeline = None


# Ã¢â€â‚¬Ã¢â€â‚¬ Public API Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def is_available() -> bool:
    """Return True if the classifier is loaded and ready to classify."""
    return _classifier_pipeline is not None


def classify_text(text: str) -> dict[str, Any]:
    """Classify text as abusive or non_abusive.

    Returns a dict with keys:
      label       Ã¢â‚¬â€ "abusive" or "non_abusive"
      abuse_score Ã¢â‚¬â€ float in [0, 1]; probability that text is abusive
      disabled    Ã¢â‚¬â€ True when classifier is not loaded (abuse_score will be 0.0)

    Never raises.  Returns a neutral disabled result on any failure so the
    moderation pipeline is never blocked by classifier errors.
    """
    if not is_available():
        return dict(_DISABLED_RESULT)

    text = text.strip()
    if not text:
        return {"label": "non_abusive", "abuse_score": 0.0, "disabled": False}

    try:
        # Pre-truncate to 512 chars; the pipeline will tokenize to max_length=128.
        output = _classifier_pipeline(text[:512], truncation=True)[0]
        label: str = output["label"]  # "non_abusive" or "abusive"
        score: float = float(output["score"])  # model confidence for that label
        abuse_score = score if label == "abusive" else 1.0 - score
        return {
            "label": label,
            "abuse_score": round(abuse_score, 4),
            "disabled": False,
        }
    except Exception:
        logger.exception("Text classifier inference failed")
        return {"label": "non_abusive", "abuse_score": 0.0, "disabled": False}
