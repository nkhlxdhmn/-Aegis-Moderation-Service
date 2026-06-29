# Configuration

Aegis Moderation runs with **zero required configuration** — copy `.env.example` to `.env` only when you want to tune behaviour.

```bash
cp .env.example .env
```

---

## All Environment Variables

### Runtime

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `local` | Label shown in logs and diagnostics |

### GPU

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0` | CUDA device index or `all` |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Docker GPU visibility |
| `VLM_DEVICE` | `cuda:0` | Device for BLIP image captioning and Whisper ASR |

### Models

| Variable | Default | Description |
|---|---|---|
| `MODEL_WARMUP` | `false` | Pre-load all models on startup (recommended for production) |
| `YOLO_CUSTOM_MODEL_PATH` | _(empty)_ | Path to a local custom YOLO weights file |
| `YOLO_OPEN_VOCAB` | `false` | Enable open-vocabulary object detection |
| `YOLO_CONFIDENCE` | `0.30` | YOLO detection confidence threshold |
| `YOLO_IOU` | `0.45` | YOLO NMS IoU threshold |
| `TEXT_CLASSIFIER_MODEL_DIR` | _(empty)_ | Local MuRIL abuse classifier weights directory |

### Caching

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_CACHE_DIR` | `./data` | Directory for FAISS embedding index |
| `CALIBRATION_FILE` | `./data/calibration.json` | YOLO confidence calibration curve |

### Image Ingestion

| Variable | Default | Description |
|---|---|---|
| `MAX_IMAGE_SIZE_MB` | `10` | Maximum image upload / download size in MB |
| `MAX_IMAGE_PIXELS` | `40000000` | Maximum image dimensions in pixels (width × height) |
| `IMAGE_DOWNLOAD_TIMEOUT_SECONDS` | `12` | Timeout for remote image URL downloads |

---

## Notes

- **No API keys required** — all models run locally.
- **No database credentials** — the platform has no persistence layer.
- **No Redis** — caching uses in-process FAISS and an in-memory hash store.
- **No Supabase / cloud** — everything runs offline after models are downloaded.

The models are downloaded lazily on first use by their respective libraries (HuggingFace Hub, Ultralytics). To pre-download before serving traffic, run:

```bash
python scripts/setup_models.py
```

Or set `MODEL_WARMUP=true` and they will be loaded at server startup.
