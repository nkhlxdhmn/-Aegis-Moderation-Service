# Aegis Moderation

[![Tests](https://github.com/your-username/aegis-moderation/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/aegis-moderation/actions/workflows/tests.yml)
[![Docker Build](https://github.com/your-username/aegis-moderation/actions/workflows/docker.yml/badge.svg)](https://github.com/your-username/aegis-moderation/actions/workflows/docker.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Aegis Moderation is a standalone AI-powered multimodal moderation platform. It analyzes images, videos, text, PDF documents, and DOCX documents, then returns a detailed report with confidence scores, extracted text, detected objects, document metadata, risk level, and an overall safety decision.

Run it with Docker and open the dashboard:

```bash
docker compose up --build
```

Then visit [http://localhost:8000](http://localhost:8000).

## Features

- React browser UI for image, video, text, PDF, and DOCX moderation.
- Versioned API endpoints under `/api/v1`.
- Moderation report with overall score, risk level, category scores, objects, OCR/extracted text, decision, and recommendation.
- PDF and DOCX validation, text extraction, metadata extraction, link detection, embedded image counting, and PII/text safety scoring.
- Downloadable JSON report and print-to-PDF report.
- No database, Redis, workers, queues, API keys, or required configuration.
- Prometheus metrics at `/metrics` and `/api/v1/metrics`.
- Docker production image with non-root user, health check, and optional GPU runtime support.
- Lazy model downloads through the underlying model libraries; model weights are not committed.

## API

- `GET /` - browser dashboard.
- `GET /api/v1/health` - service health.
- `GET /api/v1/model-health` - model loading mode and component status.
- `POST /api/v1/moderate/image` - image upload or image URL.
- `POST /api/v1/moderate/video` - video upload.
- `POST /api/v1/moderate/text` - JSON text moderation.
- `POST /api/v1/moderate/pdf` - PDF upload or PDF URL.
- `POST /api/v1/moderate/docx` - DOCX upload or DOCX URL.
- `POST /api/v1/analyze` - compatibility image endpoint.
- `POST /api/v1/moderate` - compatibility JSON image URL endpoint.
- `GET /metrics` - Prometheus scrape endpoint.

Image URL example:

```bash
curl -X POST http://localhost:8000/api/v1/moderate \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/image.jpg"}'
```

Text example:

```bash
curl -X POST http://localhost:8000/api/v1/moderate/text \
  -H "Content-Type: application/json" \
  -d '{"text":"Sample text to moderate"}'
```

## Architecture

```text
Client Browser
  |
  v
FastAPI API
  |
  v
Validation
  |
  v
OCR / Text Classifier / Vision Detection / Document Parsing
  |
  v
Rule Engine
  |
  v
JSON Moderation Report
```

## Project Structure

```text
frontend/              React browser dashboard
backend/               FastAPI application and Python backend
backend/main.py        API entrypoint served by Uvicorn
backend/pipeline/      OCR, vision, classification, and rule engine modules
backend/models/        Model registry and model metadata
scripts/               Developer and benchmark scripts
tests/                 Release and regression tests
docs/                  Project documentation
```

## Development

```bash
make install
make test
make run
```

Use `docker compose -f docker-compose.dev.yml up --build` for a live-reload development container.

## Release

The target release is `v1.0.0`. See `docs/ReleaseChecklist.md` before tagging.

Recommended GitHub topics: `ai`, `machine-learning`, `fastapi`, `computer-vision`, `ocr`, `content-moderation`, `docker`, `python`, `gpu`, `huggingface`.
