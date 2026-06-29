# Configuration

Aegis Moderation runs with no required configuration.

Optional environment variables are documented in `.env.example`:

- `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES`: optional GPU selection.
- `MODEL_WARMUP`: set to `true` only when you explicitly warm models before serving.
- `VLM_DEVICE`: preferred vision-language model device.
- `YOLO_CUSTOM_MODEL_PATH`: optional path to a local custom detector model.
- `YOLO_OPEN_VOCAB`, `YOLO_CONFIDENCE`, `YOLO_IOU`: optional detector tuning.
- `TEXT_CLASSIFIER_MODEL_DIR`: optional local text classifier path.
- `EMBEDDING_CACHE_DIR`, `CALIBRATION_FILE`: optional cache/calibration locations.
- `MAX_IMAGE_SIZE_MB`, `MAX_IMAGE_PIXELS`, `IMAGE_DOWNLOAD_TIMEOUT_SECONDS`: image ingestion limits.

No API keys, database credentials, Redis URLs, JWT secrets, or Supabase settings are required.
