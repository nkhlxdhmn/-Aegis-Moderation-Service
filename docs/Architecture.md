# Architecture

Aegis Moderation v1.0.0 is a standalone image moderation platform. It does not require a database, Redis, worker queue, or API key.

```text
Client Browser
  ↓
FastAPI API
  ↓
Validation
  ↓
OCR / Text Classifier / Vision Detection
  ↓
Rule Engine
  ↓
JSON Moderation Report
```

## Runtime Structure

- `frontend/`: browser dashboard served by FastAPI.
- `backend/`: ingestion and report-building helpers.
- `pipeline/`: OCR, vision, text safety, object detection, and rule signals.
- `models/registry.py`: model identifiers and extension points. Model weights are never committed.
- `scripts/benchmark.py`: local OCR and end-to-end benchmark runner.
- `monitoring/prometheus.yml`: Prometheus scrape example for `/metrics`.

## Data Flow

1. A user uploads an image or submits a public HTTPS image URL.
2. The backend validates content type, size, dimensions, and URL network safety.
3. The moderation pipeline extracts OCR text, object detections, text risks, and vision risks.
4. `backend.reports` normalizes the signals into the public category taxonomy.
5. The decision engine returns `SAFE`, `LOW RISK`, `MEDIUM RISK`, `HIGH RISK`, or `CRITICAL` plus a recommendation.
