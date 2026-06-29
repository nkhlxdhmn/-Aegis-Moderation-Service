"""Model warmup helpers for moderation service startup.

Loads all heavy model weights into VRAM before the FastAPI server accepts
requests, so the first moderation request doesn't pay cold-start latency.

GPU allocation:
  cuda:0 â€” all models (single-GPU default; set VLM_DEVICE env var to
            redirect BLIP/Llama to a second GPU on multi-GPU hosts).
"""

from __future__ import annotations

import logging
import os
import threading


def _vlm_device() -> str:
    return os.getenv("VLM_DEVICE", "cuda:0")

logger = logging.getLogger(__name__)

_warmup_lock = threading.Lock()
_models_loaded = False
_last_error: str | None = None


def load_nsfw() -> None:
    """Load OpenNSFW2 ViT-L on cuda:0."""
    from backend.pipeline import nsfw
    nsfw._get_state()


def load_yolo() -> None:
    """Load YOLO11x on cuda:0."""
    from backend.pipeline import object_detector
    object_detector._get_state()


def load_siglip() -> None:
    """Load SigLIP2 Large on cuda:0."""
    from backend.pipeline import clip_engine
    clip_engine._get_state()


def load_surya() -> None:
    """Load Surya OCR (primary engine). Logs outcome; no-op if package absent."""
    from backend.pipeline.surya_ocr import load_surya as _load
    ok = _load()
    if ok:
        logger.info("Surya OCR loaded")
    else:
        logger.info("Surya OCR unavailable, fallback enabled")


def load_ocr() -> None:
    """Warm the hybrid OCR engine (Surya primary + EasyOCR fallback)."""
    load_surya()
    from backend.pipeline.easyocr_engine import load_easyocr
    load_easyocr()


def load_blip2() -> None:
    """Load BLIP image-captioning-large on VLM_DEVICE (default cuda:0)."""
    from backend.pipeline import vlm_engine
    vlm_engine._get_blip()


def load_text_classifier() -> None:
    """Try to load the text abuse classifier.  No-op when MuRIL weights are absent."""
    from backend.pipeline import text_classifier
    text_classifier.load_text_classifier()


def load_ml_toxicity() -> None:
    """No-op â€” ML toxicity disabled until Dockerfile upgrades to PyTorch â‰¥ 2.6."""


def load_llama() -> None:
    """No-op â€” Llama removed. Kept for backward compatibility."""


def warmup_models() -> None:
    """Load all model weights in dependency order.

    GPU 0 models load first; GPU 1 models are loaded last so they don't
    compete for system RAM during the SigLIP/YOLO download phase.
    Errors are re-raised so the container fails health checks and is
    restarted by the orchestrator rather than silently degrading.
    """
    global _models_loaded, _last_error
    with _warmup_lock:
        try:
            logger.info("Warmup: loading OpenNSFW2 (cuda:0)")
            load_nsfw()

            logger.info("Warmup: loading SigLIP2 (cuda:0)")
            load_siglip()

            logger.info("Warmup: loading YOLO11x (cuda:0)")
            load_yolo()

            logger.info("Warmup: loading OCR engines (Surya primary + EasyOCR fallback)")
            load_ocr()

            logger.info("Warmup: loading BLIP image-captioning-large (%s)", _vlm_device())
            load_blip2()

            logger.info("Warmup: ML toxicity skipped (re-enable when torch â‰¥ 2.6)")
            load_ml_toxicity()

            logger.info("Warmup: text classifier (MuRIL hook â€” no-op if weights absent)")
            load_text_classifier()

            logger.info("Warmup: skipping Llama + Qwen (removed)")
        except Exception as exc:
            _models_loaded = False
            _last_error = str(exc)
            logger.exception("Moderation model warmup failed")
            raise
        _models_loaded = True
        _last_error = None
    logger.info("Moderation model warmup completed â€” all models on GPU")


def warmup_models_if_enabled() -> None:
    if os.getenv("MODEL_WARMUP", "").lower() == "true":
        warmup_models()


def model_status() -> str:
    if _models_loaded:
        return "loaded"
    if _last_error:
        return "error"
    return "not_loaded"


def model_status_detail() -> dict[str, str]:
    """Return per-model load state by inspecting singleton globals."""
    detail: dict[str, str] = {}
    try:
        from backend.pipeline import nsfw as _nsfw
        detail["nsfw"] = "loaded" if _nsfw._state is not None else "not_loaded"
    except Exception:
        detail["nsfw"] = "error"
    try:
        from backend.pipeline import clip_engine as _clip
        detail["siglip"] = "loaded" if _clip._state is not None else "not_loaded"
    except Exception:
        detail["siglip"] = "error"
    try:
        from backend.pipeline import object_detector as _yolo
        detail["yolo"] = "loaded" if _yolo._state is not None else "not_loaded"
    except Exception:
        detail["yolo"] = "error"
    try:
        from backend.pipeline.surya_ocr import is_available as _surya_ok
        detail["ocr_surya"] = "loaded" if _surya_ok() else "not_loaded"
    except Exception:
        detail["ocr_surya"] = "error"
    try:
        from backend.pipeline.easyocr_engine import is_available as _easyocr_ok
        detail["ocr_easyocr"] = "loaded" if _easyocr_ok() else "not_loaded"
    except Exception:
        detail["ocr_easyocr"] = "error"
    try:
        from backend.pipeline import vlm_engine as _vlm
        detail["blip"] = "loaded" if _vlm._blip_state is not None else "not_loaded"
        detail["llama"] = "disabled"  # Llama removed; rule-based decision engine active
    except Exception:
        detail["blip"] = "error"
        detail["llama"] = "error"
    try:
        from backend.pipeline import ml_toxicity as _tox
        if getattr(_tox, "_DISABLED", False):
            detail["ml_toxicity"] = "disabled"
        elif getattr(_tox, "_pipeline", None) is not None:
            detail["ml_toxicity"] = "loaded"
        else:
            detail["ml_toxicity"] = "not_loaded"
    except Exception:
        detail["ml_toxicity"] = "error"
    try:
        from backend.pipeline import text_classifier as _tc
        if _tc._classifier_disabled:
            detail["text_classifier"] = "disabled"
        elif _tc._classifier_pipeline is not None:
            detail["text_classifier"] = "loaded"
        else:
            detail["text_classifier"] = "not_loaded"
    except Exception:
        detail["text_classifier"] = "error"
    return detail


# â”€â”€ Legacy aliases (kept for backwards compat with any external callers) â”€â”€â”€â”€â”€â”€â”€
load_nudenet = load_nsfw
load_openclip = load_siglip
