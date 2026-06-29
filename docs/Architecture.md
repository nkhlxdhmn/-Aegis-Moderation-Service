# Architecture

Aegis Moderation uses a standalone, stateless architecture designed for in-process machine learning inference without the complexity of microservices or external databases.

## 1. Request Flow

1. **Ingestion**: Fastapi receives the payload (`multipart/form-data` or JSON).
2. **Validation**: `image_io.py` or `documents.py` validates limits (size, pixels, SSRF checks) and writes content to a temporary OS file.
3. **Orchestration**: The request is routed to the specific pipeline (`safety_flags.py` for images, `text_moderation.py` for text, etc.).
4. **Inference**: Content passes through the selected runtime (`CUDA -> AMD XDNA NPU through ONNX Runtime -> CPU`) and loaded model stack.
5. **Decision**: Raw model scores are aggregated by `reports.py` into a final `risk_level` and `decision`.
6. **Cleanup**: Temporary files are deleted.

## 2. Models Used

| Component | Model | Purpose |
|---|---|---|
| **Vision (NSFW)** | `Falconsai/nsfw_image_detection` | Detect explicit/suggestive content |
| **Object Detection** | `YOLO11x` (Ultralytics) | Detect weapons, drugs, gore |
| **Vision Embeddings** | `google/siglip2-large-patch16-384` | Zero-shot safety classification |
| **OCR** | `Surya OCR` | Extract text from images |
| **Image Captioning** | `Salesforce/blip-image-captioning-large` | Provide visual context for text models |
| **Text Moderation** | `Detoxify` (Multilingual) | Detect toxicity, insults, identity hate |
| **Language ID** | `FastText` | Detect content language |
| **Audio** | `Whisper` | Video transcription |

## 3. The 11-Stage Image Pipeline

1. Validation & Input Normalization
2. Hash Caching (In-memory FAISS similarity)
3. NSFW & Gore Classification
4. Object Detection (YOLO)
5. SigLIP Zero-Shot Safety
6. OCR & Text Extraction
7. BLIP Image Captioning
8. Text Moderation (on OCR & Captions)
9. PII & Document Detection
10. Rule Engine Aggregation
11. Uncertainty Estimation

## 4. State & Monitoring

The application uses an in-process singleton (`AegisMonitor`) in `backend/monitor.py`. This tracks:
- System resources (CPU, Memory, Disk, GPU) via `psutil` / `pynvml`
- Ring-buffers (last 1000) for requests, errors, and latency
- 10-second history buckets for throughput graphing

No external database (like PostgreSQL or Redis) is required.

## 5. Runtime Selection

Hardware detection lives in `core/runtime/`:

| Priority | Runtime | Backend |
|---|---|---|
| 1 | NVIDIA CUDA | PyTorch CUDA / optional ONNX CUDA provider |
| 2 | AMD XDNA NPU | ONNX Runtime `VitisAIExecutionProvider` |
| 3 | CPU | CPU fallback |

The NPU backend is optional. If ONNX Runtime or the AMD execution provider is unavailable, the service continues on CPU. Runtime details are exposed at `GET /api/runtime/status`.
