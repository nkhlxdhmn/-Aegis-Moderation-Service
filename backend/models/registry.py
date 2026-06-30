"""Model identifiers and cache locations for Aegis Moderation.

Model files are intentionally not bundled in the repository. The underlying model
libraries download them into the configured cache on first use, or operators can run
`python model_warmup.py` before serving traffic.
"""

from __future__ import annotations

OCR_MODEL = "surya-ocr"
SURYA_OCR_MODEL = "surya-ocr/default"
TEXT_MODEL = "detoxify/unbiased"
VISION_MODEL = "qwen-vl/default"
NSFW_MODEL = "opennsfw2/default"
YOLO_MODEL = "ultralytics/yolov8n"
WHISPER_MODEL = "optional/whisper"
CUSTOM_VISION_MODEL_ENV = "YOLO_CUSTOM_MODEL_PATH"

MODEL_REGISTRY = {
    "ocr": OCR_MODEL,
    "surya_ocr": SURYA_OCR_MODEL,
    "text": TEXT_MODEL,
    "vision": VISION_MODEL,
    "nsfw": NSFW_MODEL,
    "yolo": YOLO_MODEL,
    "whisper": WHISPER_MODEL,
}
