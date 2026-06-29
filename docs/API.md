# API

Base URL: `http://localhost:8000`

## Health

`GET /api/v1/health` returns service status.

`GET /api/v1/model-health` reports lazy-loading model readiness.

`GET /api/v1/metrics` and `GET /metrics` expose Prometheus metrics.

## Moderation Endpoints

### `POST /api/v1/moderate/image`

Multipart form endpoint for image uploads or public HTTPS image URLs.

Fields:

- `file`: optional image upload.
- `image_url`: optional public HTTPS image URL.
- `caption`: optional context string.

Provide exactly one of `file` or `image_url`.

### `POST /api/v1/moderate/video`

Multipart form endpoint for video uploads.

Fields:

- `file`: required video upload.
- `caption`: optional context string.

### `POST /api/v1/moderate/text`

JSON endpoint for plain text moderation.

```json
{
  "text": "Sample text to moderate"
}
```

### `POST /api/v1/moderate/pdf`

Multipart form endpoint for PDF upload or public HTTPS PDF URL.

Fields:

- `file`: optional `.pdf` upload.
- `document_url`: optional public HTTPS `.pdf` URL.

Provide exactly one of `file` or `document_url`.

### `POST /api/v1/moderate/docx`

Multipart form endpoint for DOCX upload or public HTTPS DOCX URL.

Fields:

- `file`: optional `.docx` upload.
- `document_url`: optional public HTTPS `.docx` URL.

Provide exactly one of `file` or `document_url`.

## Compatibility Endpoints

- `POST /api/v1/analyze`: previous browser image endpoint.
- `POST /api/v1/moderate`: JSON image URL endpoint.

## Response Shape

```json
{
  "overall_score": 83.2,
  "risk_level": "HIGH RISK",
  "decision": "Review Required",
  "recommendation": "Human Review",
  "categories": {
    "weapons": 74.1,
    "adult_content": 0.4,
    "pii_detection": 0
  },
  "objects": ["knife", "person"],
  "ocr_text": "...",
  "extracted_text_preview": "...",
  "document": {
    "file_info": {
      "filename": "sample.pdf",
      "file_type": "PDF",
      "file_size_bytes": 12345
    },
    "page_count": 2,
    "processing_time_seconds": 0.12,
    "metadata": {},
    "links": [],
    "embedded_images": 0
  }
}
```

## OpenAPI

FastAPI serves OpenAPI at `/api/openapi.json` and interactive docs at `/api/docs`.
