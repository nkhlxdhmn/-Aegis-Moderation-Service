# API Reference

Base URL: `http://localhost:8000`

Interactive Swagger UI: `http://localhost:8000/api/docs`
ReDoc: `http://localhost:8000/api/redoc`
OpenAPI JSON: `http://localhost:8000/api/openapi.json`

---

## Health Endpoints

### `GET /api/v1/health`

Service liveness check.

**Response**
```json
{
  "status": "ok",
  "service": "Aegis Moderation",
  "version": "1.0.0",
  "mode": "standalone"
}
```

### `GET /api/v1/model-health`

Reports model readiness without triggering downloads.

**Response**
```json
{
  "status": "ready",
  "mode": "lazy-load",
  "message": "Models are loaded or downloaded by the pipeline on first analysis.",
  "components": ["OCR", "Text Classifier", "Vision Models", "Rule Engine", "Document Parser"]
}
```

### `GET /api/v1/metrics` · `GET /metrics`

Prometheus metrics in text format. Returns plain text if `prometheus_client` is not installed.

---

## Moderation Endpoints

All moderation endpoints return the same **Report** schema (see below).

---

### `POST /api/v1/moderate/image`

Analyze an image for NSFW content, violence, weapons, hate symbols, QR codes, OCR text, and PII.

**Request** — multipart/form-data. Provide exactly one of `file` or `image_url`.

| Field | Type | Description |
|---|---|---|
| `file` | file upload | JPEG, PNG, WEBP, or GIF — max 10 MB |
| `image_url` | string | Public HTTPS image URL |
| `caption` | string (optional) | Context text to improve classification |

**curl — upload**
```bash
curl -X POST http://localhost:8000/api/v1/moderate/image \
  -F "file=@photo.jpg"
```

**curl — URL**
```bash
curl -X POST http://localhost:8000/api/v1/moderate/image \
  -F "image_url=https://example.com/photo.jpg"
```

---

### `POST /api/v1/moderate/video`

Analyze a video file (frame-by-frame + audio transcription).

**Request** — multipart/form-data.

| Field | Type | Description |
|---|---|---|
| `file` | file upload | MP4, MOV, MKV, WebM — max 250 MB |
| `caption` | string (optional) | Context text |

The response includes an additional `video` key:
```json
{
  "video": {
    "frame_count": 42,
    "unsafe_frame_count": 0,
    "max_consecutive_unsafe": 0,
    "transcript": ""
  }
}
```

---

### `POST /api/v1/moderate/text`

Analyze plain text for toxicity, hate speech, spam, PII, and language.

**Request** — `application/json`

```json
{ "text": "Your text content here" }
```

| Field | Constraints |
|---|---|
| `text` | 1–100 000 characters |

The response includes `extracted_text_preview` containing the first 2000 characters of the submitted text.

---

### `POST /api/v1/moderate/pdf`

Analyze a PDF document. Extracts text, detects links, checks for phishing/malware, and scans for PII.

**Request** — multipart/form-data. Provide exactly one of `file` or `document_url`.

| Field | Type | Description |
|---|---|---|
| `file` | file upload | PDF — max 25 MB, max 80 pages |
| `document_url` | string | Public HTTPS PDF URL |

---

### `POST /api/v1/moderate/docx`

Analyze a DOCX document. Extracts paragraphs and table text, checks for embedded executables and zip bombs.

**Request** — multipart/form-data. Provide exactly one of `file` or `document_url`.

| Field | Type | Description |
|---|---|---|
| `file` | file upload | DOCX — max 25 MB |
| `document_url` | string | Public HTTPS DOCX URL |

---

## Report Schema

All moderation endpoints return the same JSON structure:

```json
{
  "overall_score": 4.2,
  "risk_level": "SAFE",
  "decision": "Accept",
  "recommendation": "Allow",
  "categories": {
    "adult_content": 0.0,
    "nudity": 0.0,
    "suggestive_content": 0.0,
    "violence": 0.0,
    "graphic_violence": 0.0,
    "weapons": 0.0,
    "drugs": 0.0,
    "blood": 0.0,
    "medical_content": 0.0,
    "political_propaganda": 0.0,
    "religious_extremism": 0.0,
    "hate_speech": 0.0,
    "hate_symbol": 0.0,
    "toxic_text": 0.0,
    "spam": 0.0,
    "scam": 0.0,
    "misinformation": 0.0,
    "self_harm": 0.0,
    "child_safety_risk": 0.0,
    "pii_detection": 0.0,
    "qr_code": 0.0,
    "document": 0.0
  },
  "category_labels": {
    "adult_content": "Adult Content",
    "violence": "Violence"
  },
  "objects": ["person", "car"],
  "ocr_text": "Extracted OCR text here",
  "caption": "BLIP generated caption",
  "image_hash": "a1b2c3d4",
  "model_versions": {
    "nsfw": "Falconsai/nsfw_image_detection",
    "siglip": "google/siglip2-large-patch16-384"
  },
  "content_type": "image",
  "error": null
}
```

### Risk Levels

| `risk_level` | `overall_score` | `decision` |
|---|---|---|
| `SAFE` | 0–19 | Accept |
| `LOW RISK` | 20–39 | Accept (monitor) |
| `MEDIUM RISK` | 40–69 | Review Required |
| `HIGH RISK` | 70–89 | Review Required |
| `CRITICAL` | ≥90 | Reject |

Child safety risk ≥80 or adult content ≥90 immediately triggers `CRITICAL` regardless of overall score.

---

## Error Responses

```json
{ "detail": "Provide either an uploaded image or an image URL." }
```

| Status | Cause |
|---|---|
| `400` | Validation error (wrong extension, empty file, private URL, size exceeded) |
| `500` | Internal pipeline error |
