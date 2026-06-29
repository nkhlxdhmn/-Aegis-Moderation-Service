"""Safe model warmup helpers for moderation service startup.

Each model is loaded independently so one missing package, unavailable GPU, or
bad weight file cannot prevent the FastAPI application from starting. The
module keeps a small in-memory status map that powers the dashboard and health
APIs.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Literal

logger = logging.getLogger(__name__)

ModelState = Literal["loaded", "failed", "disabled", "not_loaded", "lazy"]

MODEL_KEYS: tuple[str, ...] = (
    "nsfw",
    "siglip",
    "yolo",
    "ocr_surya",
    "blip",
    "llama",
    "text_classifier",
)

_warmup_lock = threading.Lock()
_models_loaded = False
_last_error: str | None = None
_model_status: dict[str, ModelState] = {key: "not_loaded" for key in MODEL_KEYS}
_model_errors: dict[str, str] = {}


def _vlm_device() -> str:
    return os.getenv("VLM_DEVICE", "cuda:0")


def _record_model_load(name: str, duration: float) -> None:
    try:
        from backend.monitor import monitor

        monitor.record_model_load(name, duration)
    except Exception:
        logger.debug("Unable to record model load timing for %s", name, exc_info=True)


def _warmup_enabled() -> bool:
    return os.getenv("MODEL_WARMUP", "").lower() == "true"


def _set_status(key: str, state: ModelState, error: str | None = None) -> None:
    _model_status[key] = state
    if error:
        _model_errors[key] = error
    else:
        _model_errors.pop(key, None)


def safe_load(name: str, loader: Callable[[], object]) -> bool:
    """Load one model and return whether the loader completed without raising."""
    started = time.perf_counter()
    try:
        logger.info("Loading %s", name)
        loader()
        duration = time.perf_counter() - started
        _record_model_load(name, duration)
        logger.info("%s loaded successfully", name)
        return True
    except Exception as exc:
        duration = time.perf_counter() - started
        _record_model_load(name, duration)
        logger.exception("%s failed: %s", name, exc)
        return False


def load_nsfw() -> None:
    """Load OpenNSFW2 ViT-L."""
    from backend.pipeline import nsfw

    nsfw._get_state()


def load_yolo() -> None:
    """Load YOLO11x."""
    from backend.pipeline import object_detector

    object_detector._get_state()


def load_siglip() -> None:
    """Load SigLIP2 Large."""
    from backend.pipeline import clip_engine

    clip_engine._get_state()


def load_surya() -> None:
    """Load Surya OCR, raising if the primary OCR engine is unavailable."""
    from backend.pipeline.surya_ocr import load_surya as _load

    if not _load():
        raise RuntimeError("Surya OCR unavailable")


def load_ocr() -> None:
    """Warm the Surya OCR engine."""
    load_surya()


def load_blip() -> None:
    """Load BLIP image captioning."""
    from backend.pipeline import vlm_engine

    vlm_engine._get_blip()


def load_blip2() -> None:
    """Backward-compatible alias for BLIP warmup."""
    load_blip()


def load_llama() -> None:
    """Load Llama reasoning model."""
    from backend.pipeline import vlm_engine

    vlm_engine._get_llama()


def load_text_classifier() -> None:
    """Try to load the text abuse classifier."""
    from backend.pipeline import text_classifier

    text_classifier.load_text_classifier()


def _classifier_state() -> ModelState:
    try:
        from backend.pipeline import text_classifier

        if text_classifier.is_available():
            return "loaded"
        if getattr(text_classifier, "_classifier_disabled", False):
            return "disabled"
    except Exception:
        logger.debug("Unable to inspect text classifier state", exc_info=True)
        return "failed"
    return "not_loaded"


def _live_state_from_singletons(key: str) -> ModelState:
    """Inspect already-imported singleton state without forcing downloads."""
    try:
        if key == "nsfw":
            from backend.pipeline import nsfw

            return "loaded" if nsfw._state is not None else "not_loaded"
        if key == "siglip":
            from backend.pipeline import clip_engine

            return "loaded" if clip_engine._state is not None else "not_loaded"
        if key == "yolo":
            from backend.pipeline import object_detector

            return "loaded" if object_detector._state is not None else "not_loaded"
        if key == "ocr_surya":
            from backend.pipeline.surya_ocr import is_available

            return "loaded" if is_available() else "not_loaded"
        if key == "blip":
            from backend.pipeline import vlm_engine

            return "loaded" if vlm_engine._blip_state is not None else "not_loaded"
        if key == "llama":
            from backend.pipeline import vlm_engine

            return "loaded" if vlm_engine._llama_state is not None else "not_loaded"
        if key == "text_classifier":
            return _classifier_state()
    except Exception:
        logger.debug("Unable to inspect %s state", key, exc_info=True)
        return "failed"
    return "not_loaded"


def warmup_models() -> dict[str, ModelState]:
    """Load all models safely and return per-model load status."""
    global _models_loaded, _last_error

    with _warmup_lock:
        _models_loaded = False
        _last_error = None

        load_plan: tuple[tuple[str, str, Callable[[], object]], ...] = (
            ("nsfw", "NSFW", load_nsfw),
            ("siglip", "SigLIP", load_siglip),
            ("yolo", "YOLO", load_yolo),
            ("ocr_surya", "Surya OCR", load_surya),
            ("blip", f"BLIP ({_vlm_device()})", load_blip),
            ("llama", f"Llama ({_vlm_device()})", load_llama),
            ("text_classifier", "Text Classifier", load_text_classifier),
        )

        for key, label, loader in load_plan:
            _set_status(key, "not_loaded")
            ok = safe_load(label, loader)
            if ok:
                status = _classifier_state() if key == "text_classifier" else "loaded"
                _set_status(key, status)
            else:
                _set_status(key, "failed", f"{label} failed during warmup")

        _models_loaded = True
        failed = [key for key, state in _model_status.items() if state == "failed"]
        _last_error = ", ".join(failed) if failed else None

        if failed:
            logger.warning("Model warmup completed with failures: %s", ", ".join(failed))
        else:
            logger.info("Model warmup completed successfully")

        return dict(_model_status)


def warmup_models_if_enabled() -> dict[str, ModelState]:
    if _warmup_enabled():
        return warmup_models()

    logger.info("MODEL_WARMUP is disabled; models will load lazily")
    return dict(_model_status)


def model_status() -> str:
    if any(state == "failed" for state in _model_status.values()):
        return "failed"
    if _models_loaded:
        return "loaded"
    if _last_error:
        return "failed"
    if not _warmup_enabled():
        return "lazy"
    return "not_loaded"


def model_status_detail() -> dict[str, ModelState]:
    """Return per-model load state for the public API and dashboard."""
    detail: dict[str, ModelState] = {}
    warmup_disabled = not _warmup_enabled()

    for key in MODEL_KEYS:
        tracked = _model_status.get(key, "not_loaded")
        live = _live_state_from_singletons(key)

        if live == "loaded":
            detail[key] = "loaded"
        elif tracked == "failed" or live == "failed":
            detail[key] = "failed"
        elif live == "disabled" or tracked == "disabled":
            detail[key] = "disabled"
        elif warmup_disabled:
            detail[key] = "lazy"
        else:
            detail[key] = tracked

    return detail


def model_errors() -> dict[str, str]:
    return dict(_model_errors)


# Legacy aliases kept for backwards compatibility with external callers.
load_nudenet = load_nsfw
load_openclip = load_siglip
