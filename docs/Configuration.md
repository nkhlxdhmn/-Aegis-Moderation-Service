# Configuration

Aegis Moderation requires zero configuration to run locally. However, you can tune its behavior using environment variables in a `.env` file or `docker-compose.yml`.

## Core Variables

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `local` | Label used in logs (`local`, `production`, `development`). |
| `MODEL_WARMUP` | `false` | If `true`, loads all models safely at startup before accepting traffic. Keep `false` for local Docker unless you have enough RAM/VRAM for the full stack. |

## GPU Configuration

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_VISIBLE_DEVICES` | `all` | Controls which GPUs Docker exposes to the container. |
| `CUDA_VISIBLE_DEVICES` | `0` | Controls which GPUs PyTorch sees. e.g., `0,1`. |
| `AEGIS_CUDA_DEVICE` | `cuda:0` | Preferred CUDA device for runtime selection. |
| `VLM_DEVICE` | `cuda:0` | Which device to use for heavy models (BLIP/Whisper). On multi-GPU hosts, set to `cuda:1` to split memory. |

## Optional NPU Configuration

AMD XDNA NPU acceleration is detected through ONNX Runtime when `VitisAIExecutionProvider` is available. It is not required for startup.

| Variable | Default | Description |
|---|---|---|
| `VITISAI_CONFIG` | (empty) | Optional VitisAI provider config file path. |
| `RYZEN_AI_CONFIG` | (empty) | Alternate config file path used when `VITISAI_CONFIG` is unset. |
| `VITISAI_PROVIDER_OPTIONS` | (empty) | Optional JSON object merged into VitisAI provider options. |

## Input Limits

| Variable | Default | Description |
|---|---|---|
| `MAX_IMAGE_SIZE_MB` | `10` | Maximum allowed file size for image uploads. |
| `MAX_IMAGE_PIXELS` | `40000000` | Maximum pixel count (width × height). Rejects huge images that cause OOM. |
| `IMAGE_DOWNLOAD_TIMEOUT_SECONDS` | `12` | Network timeout when downloading images/documents via URL. |

## Model Tuning

| Variable | Default | Description |
|---|---|---|
| `YOLO_CONFIDENCE` | `0.30` | Minimum confidence threshold for YOLO11x object detection. |
| `YOLO_IOU` | `0.45` | Intersection-over-Union threshold for YOLO NMS. |
| `YOLO_CUSTOM_MODEL_PATH` | (empty) | Path to a custom fine-tuned YOLO weights file (`.pt`). |
| `EMBEDDING_CACHE_DIR` | `./data` | Where FAISS similarity index is saved. |
| `CALIBRATION_FILE` | `./data/calibration.json` | JSON file for overriding rule engine thresholds. |
