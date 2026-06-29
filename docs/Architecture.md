# Architecture

Aegis Moderation v1.0.0 is a standalone multimodal moderation platform. It does not require a database, Redis, worker queue, or API key.

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
OCR / Text Classifier / Vision Models / Document Parsers
  |
  v
Rule Engine
  |
  v
JSON Moderation Report
```

## Runtime Structure

- `frontend/`: React browser dashboard served by FastAPI.
- `backend/`: FastAPI entrypoint, ingestion helpers, document parsers, and report builders.
- `backend/pipeline/`: OCR, vision, text safety, object detection, and rule signals.
- `backend/models/registry.py`: model identifiers and extension points. Model weights are never committed.
- `scripts/benchmark.py`: local OCR and end-to-end benchmark runner.
- `monitoring/prometheus.yml`: Prometheus scrape example for `/metrics`.

## Data Flow

1. A user submits an image, video, text, PDF, or DOCX file from the React UI.
2. The backend validates content type, size, URL network safety, and document safety limits.
3. The relevant pipeline extracts OCR text, document text, object detections, text risks, and vision risks.
4. `backend.reports` normalizes the signals into the public category taxonomy.
5. The decision engine returns `SAFE`, `LOW RISK`, `MEDIUM RISK`, `HIGH RISK`, or `CRITICAL` plus a recommendation.

## Document Flow

```text
PDF/DOCX
  |
  v
Validate File or URL
  |
  v
Extract Metadata and Text
  |
  v
Detect Links, Embedded Images, PII, Spam, Scam, and Toxic Text
  |
  v
Rule Engine
  |
  v
Document Moderation Report
```
