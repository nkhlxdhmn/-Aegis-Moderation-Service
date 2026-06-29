"""YOLOv11x object detection for the MyItihas moderation pipeline.

Model: yolo11x.pt (Ultralytics YOLO11 Extra-Large, COCO-pretrained).
Device: cuda:0.

IMPORTANT — COCO class constraint:
  Only COCO-80 classes (person, knife, etc.) are reliably detected by the
  default YOLO11x model. Heritage-specific labels (temple, idol, statue,
  religious_symbol, festival, diya) do NOT exist in COCO-80 and must NOT be
  assumed — doing so produces hallucinated detections.
  Heritage context is handled by SigLIP2 semantic prompts instead.

Custom model support:
  Set YOLO_CUSTOM_MODEL_PATH to a fine-tuned model path that includes
  heritage classes. The loader will prefer it automatically.

Open-vocabulary option:
  Set YOLO_OPEN_VOCAB=true to load yolov8x-worldv2 with custom class names
  without retraining (uses YOLO-World zero-shot detection).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
YOLO_MODEL_DEFAULT = "yolo11x.pt"
DEVICE = "cuda:0"

# Path to a custom-trained MyItihas YOLO model (overrides default when set)
CUSTOM_MODEL_PATH: str = os.getenv("YOLO_CUSTOM_MODEL_PATH", "")

# Enable YOLO-World open-vocabulary mode for custom class names without retraining
USE_OPEN_VOCAB: bool = os.getenv("YOLO_OPEN_VOCAB", "false").lower() == "true"

# ── COCO-safe MyItihas classes ─────────────────────────────────────────────────
# Only classes that actually exist in COCO-80 or YOLO11x default weights.
# Heritage classes (temple/idol/statue/religious_symbol/festival/diya) are
# intentionally absent — they do not exist in COCO and produce hallucinations.
MYITIHAS_CLASSES: list[str] = [
    "person",
    "child",   # available in YOLO-World / custom fine-tunes
    "weapon",
    "fire",
    "crowd",   # synthetic — inferred from ≥5 concurrent person detections
]

# ── COCO class → MyItihas semantic mapping ─────────────────────────────────────
COCO_TO_MYITIHAS: dict[str, str] = {
    "person": "person",
    "knife": "weapon",
    "scissors": "weapon",
    "gun": "weapon",
    "pistol": "weapon",
    "rifle": "weapon",
    "sword": "weapon",
    # fire / crowd are handled by custom models or synthetic inference
}

# Minimum detection confidence — lower than default to catch partially obscured objects
CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE", "0.30"))
# IOU threshold for NMS
IOU_THRESHOLD: float = float(os.getenv("YOLO_IOU", "0.45"))


class ModelInferenceError(RuntimeError):
    """Raised when YOLO cannot load or process an image."""


@dataclass
class _YOLOState:
    model: Any
    device: str
    torch: Any
    is_custom: bool
    is_open_vocab: bool


_state: _YOLOState | None = None
_state_lock = threading.Lock()


def _select_device() -> str:
    try:
        import torch
        return DEVICE if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model(device: str) -> tuple[Any, bool, bool]:
    """Load the best available YOLO model and return (model, is_custom, is_open_vocab)."""
    from ultralytics import YOLO

    if CUSTOM_MODEL_PATH and os.path.isfile(CUSTOM_MODEL_PATH):
        logger.info("Loading custom MyItihas YOLO model: %s", CUSTOM_MODEL_PATH)
        model = YOLO(CUSTOM_MODEL_PATH)
        model.to(device)
        return model, True, False

    if USE_OPEN_VOCAB:
        logger.info("Loading YOLO-World open-vocabulary model on %s", device)
        model = YOLO("yolov8x-worldv2.pt")
        model.set_classes(MYITIHAS_CLASSES)
        model.to(device)
        return model, False, True

    logger.info("Loading YOLO11x (COCO pretrained) on %s", device)
    model = YOLO(YOLO_MODEL_DEFAULT)
    model.to(device)
    return model, False, False


def _get_state() -> _YOLOState:
    global _state
    if _state is not None:
        return _state

    with _state_lock:
        if _state is not None:
            return _state

        try:
            import torch
            device = _select_device()
            model, is_custom, is_open_vocab = _load_model(device)
            _state = _YOLOState(
                model=model,
                device=device,
                torch=torch,
                is_custom=is_custom,
                is_open_vocab=is_open_vocab,
            )
        except Exception as exc:
            logger.exception("Failed to load YOLO model")
            raise ModelInferenceError("YOLO model failed to load") from exc

        logger.info(
            "YOLO model loaded on %s (custom=%s, open_vocab=%s)",
            _state.device, _state.is_custom, _state.is_open_vocab,
        )
    return _state


def _remap_class(class_name: str) -> str:
    """Map a COCO class name to a MyItihas semantic class where applicable."""
    return COCO_TO_MYITIHAS.get(class_name.lower(), class_name.lower())


def _infer_crowd(detections: list[dict]) -> list[dict]:
    """Add a synthetic 'crowd' detection when many persons appear in one frame."""
    person_count = sum(1 for d in detections if d.get("class") == "person")
    if person_count >= 5:
        avg_conf = sum(
            d["confidence"] for d in detections if d.get("class") == "person"
        ) / person_count
        detections.append({"class": "crowd", "confidence": round(avg_conf, 4)})
    return detections


def _parse_results(results: Any, state: _YOLOState) -> list[dict]:
    detections: list[dict] = []
    if results is None:
        return detections

    for result in (results if isinstance(results, list) else [results]):
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        cls_vals = getattr(boxes, "cls", None)
        conf_vals = getattr(boxes, "conf", None)
        if cls_vals is None or conf_vals is None:
            continue

        names: dict | list | None = getattr(result, "names", None)
        if names is None:
            names = getattr(getattr(result, "model", None), "names", None)

        for cls_t, conf_t in zip(cls_vals, conf_vals):
            try:
                cls_id = int(cls_t.item() if hasattr(cls_t, "item") else float(cls_t))
                conf = float(conf_t.item() if hasattr(conf_t, "item") else float(conf_t))
            except Exception:
                continue

            if conf < CONFIDENCE_THRESHOLD:
                continue

            if isinstance(names, dict):
                raw_name = str(names.get(cls_id, cls_id))
            elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                raw_name = str(names[cls_id])
            else:
                raw_name = str(cls_id)

            # Use custom class names directly if from a custom or open-vocab model
            mapped = raw_name if (state.is_custom or state.is_open_vocab) else _remap_class(raw_name)
            detections.append({"class": mapped, "confidence": round(conf, 4)})

    return _infer_crowd(detections)


def detect_objects(image_path: str) -> list[dict]:
    """Return YOLO11x object detections for an image.

    Each detection: {"class": str, "confidence": float [0,1]}
    """
    logger.info("YOLO11x object detection started")
    state = _get_state()

    try:
        with state.torch.inference_mode():
            results = state.model(
                image_path,
                device=state.device,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
            )
    except Exception as exc:
        if state.device != "cpu":
            logger.exception("YOLO CUDA inference failed; retrying on CPU")
            try:
                if state.torch.cuda.is_available():
                    state.torch.cuda.empty_cache()
                with state.torch.inference_mode():
                    results = state.model(
                        image_path,
                        device="cpu",
                        conf=CONFIDENCE_THRESHOLD,
                        iou=IOU_THRESHOLD,
                        verbose=False,
                    )
            except Exception as retry_exc:
                logger.exception("YOLO CPU fallback inference failed")
                raise ModelInferenceError("YOLO object detection failed") from retry_exc
        else:
            logger.exception("YOLO object detection failed")
            raise ModelInferenceError("YOLO object detection failed") from exc

    detections = _parse_results(results, state)
    logger.info("YOLO11x detection completed: %d objects", len(detections))
    return detections
