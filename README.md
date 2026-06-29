# Aegis Moderation

[![Tests](https://github.com/your-org/aegis-moderation/actions/workflows/tests.yml/badge.svg)](https://github.com/your-org/aegis-moderation/actions/workflows/tests.yml)
[![Docker Build](https://github.com/your-org/aegis-moderation/actions/workflows/docker.yml/badge.svg)](https://github.com/your-org/aegis-moderation/actions/workflows/docker.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Standalone AI-powered multimodal content moderation platform.**

Analyze images, videos, text, PDFs, and DOCX files with a full pipeline of locally-run vision models, OCR, language detection, and a rule-based decision engine — all accessible through a browser dashboard served directly by the API.

**No cloud. No database. No Redis. No auth service.** Just `docker compose up`.

---

## Features

| Content Type | What gets analyzed |
|---|---|
| **Image** | NSFW, violence, weapons, hate symbols, QR codes, OCR text, PII, cultural context |
| **Video** | Per-frame vision analysis + Whisper audio transcription |
| **Text** | Toxicity, hate speech, spam, PII, language detection |
| **PDF** | Text extraction, link analysis, phishing / malware detection |
| **DOCX** | Paragraph + table text, embedded image count, PII |

**AI models:** NSFW classifier · YOLO11x object detector · SigLIP2 vision encoder · Surya + EasyOCR · BLIP image captioning · Detoxify multilingual toxicity · FastText language ID.

---

## Quick Start

### Docker (GPU — recommended for production)

```bash
docker compose up --build
```

### Docker (CPU — local dev, no GPU required)

```bash
docker compose -f docker-compose.dev.yml up --build
```

The dashboard opens at **http://localhost:8000** automatically.

### Local Python

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Pre-download all model weights (also happens lazily on first request)
python scripts/setup_models.py

uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## REST API

Full interactive docs at **http://localhost:8000/api/docs** (Swagger UI).

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/moderate/image` | Upload file or pass `image_url` form field |
| `POST` | `/api/v1/moderate/video` | Upload video file |
| `POST` | `/api/v1/moderate/text` | JSON `{ "text": "..." }` |
| `POST` | `/api/v1/moderate/pdf` | Upload file or pass `document_url` form field |
| `POST` | `/api/v1/moderate/docx` | Upload file or pass `document_url` form field |
| `GET`  | `/api/v1/health` | Service liveness |
| `GET`  | `/api/v1/model-health` | Model readiness status |
| `GET`  | `/api/v1/metrics` | Prometheus metrics |

### Example — image URL

```bash
curl -X POST http://localhost:8000/api/v1/moderate/image \
  -F "image_url=https://example.com/photo.jpg"
```

### Example — text

```bash
curl -X POST http://localhost:8000/api/v1/moderate/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

### Response schema (condensed)

```json
{
  "overall_score": 4.2,
  "risk_level": "SAFE",
  "decision": "Accept",
  "recommendation": "Allow",
  "categories": {
    "adult_content": 0.0,
    "violence": 0.0,
    "hate_speech": 0.0
  },
  "objects": ["person", "car"],
  "ocr_text": "",
  "content_type": "image"
}
```

---

## Configuration

```bash
cp .env.example .env   # then edit as needed
```

| Variable | Default | Description |
|---|---|---|
| `MAX_IMAGE_SIZE_MB` | `10` | Image upload size cap |
| `MAX_IMAGE_PIXELS` | `40000000` | Max image pixel count |
| `IMAGE_DOWNLOAD_TIMEOUT_SECONDS` | `12` | Timeout for remote image downloads |
| `MODEL_WARMUP` | `false` | Pre-load all models at startup |
| `VLM_DEVICE` | `cuda:0` | Device for BLIP / Whisper |
| `CUDA_VISIBLE_DEVICES` | `0` | NVIDIA GPU selection |

See [docs/Configuration.md](docs/Configuration.md) for the full list.

---

## Project Structure

```
backend/
  main.py               FastAPI app + REST endpoints
  image_io.py           Image upload + SSRF-safe URL download
  documents.py          PDF and DOCX ingestion
  reports.py            Report normalisation + decision engine
  model_warmup.py       Eagerly pre-load model weights
  validate_models.py    Pre-flight model validation checklist
  pipeline/
    safety_flags.py     Image pipeline orchestration (11 stages)
    text_moderation.py  Text moderation pipeline
    video_moderation.py Video frame + transcript pipeline
    nsfw.py             NSFW / explicit classifier
    clip_engine.py      SigLIP2 vision embeddings
    object_detector.py  YOLO11x object detection
    ocr.py              Hybrid OCR (Surya + EasyOCR)
    vlm_engine.py       BLIP image captioning
    decision_engine.py  Rule-based decision tree
    ...
frontend/
  index.html            Standalone browser dashboard (no build step)
scripts/
  setup_models.py       Download all model weights
  warmup.py             Pre-warm models before accepting traffic
  benchmark.py          Latency benchmarking
tests/
  test_main.py          FastAPI endpoint tests
  test_standalone_report.py  Report normalisation tests
  ...
```

---

## Security

- **SSRF protection** — only public HTTPS URLs accepted; RFC-1918 / loopback IPs blocked at DNS-resolution time
- **Path traversal** — all uploads written to OS temp files via `tempfile`, never to user-controlled paths
- **MIME validation** — content-type checked for both uploads and URL-downloaded content
- **Size limits** — configurable; 10 MB default for images, 25 MB for documents
- **Zip bomb protection** — DOCX decompression ratio capped at 100×, max 80 MB uncompressed
- **Executable detection** — DOCX files containing embedded `.exe` / `.dll` / `.ps1` etc. are rejected immediately
- **Security headers** — `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy` on every response

See [SECURITY.md](SECURITY.md) for the responsible disclosure policy.

---

## Development

```bash
# Run core tests (no GPU or model downloads required)
python -m pytest tests/test_main.py tests/test_standalone_report.py -v

# Lint
ruff check .
black --check .

# Format
black .

# Validate models after install
python backend/validate_models.py --skip-warmup

# Benchmark OCR + end-to-end latency
python scripts/benchmark.py --image path/to/image.jpg --ocr-only
```

---

## Docs

| Document | Description |
|---|---|
| [docs/API.md](docs/API.md) | Full API reference with curl examples |
| [docs/Architecture.md](docs/Architecture.md) | Pipeline stages and model architecture |
| [docs/Configuration.md](docs/Configuration.md) | All environment variables |
| [docs/Deployment.md](docs/Deployment.md) | Docker and cloud deployment guides |
| [docs/Performance.md](docs/Performance.md) | Latency tuning and GPU allocation |
| [docs/Security.md](docs/Security.md) | Threat model and security controls |
| [docs/Roadmap.md](docs/Roadmap.md) | Planned features |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

---

## License

[MIT](LICENSE) — © Aegis Moderation Contributors
