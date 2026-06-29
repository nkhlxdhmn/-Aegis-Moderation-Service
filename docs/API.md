# Aegis Moderation API Reference

The REST API allows external systems to submit content for moderation. All endpoints return JSON.
Interactive Swagger UI is available at `http://localhost:8000/api/docs`.

## Endpoints

### 1. `POST /api/v1/moderate/image`
Moderates an image file or URL.

**Request (Multipart Form):**
- `file`: (Optional) The image file to upload.
- `image_url`: (Optional) A public HTTPS URL to an image.
- `caption`: (Optional) Contextual text to improve analysis accuracy.

*Note: Provide exactly one of `file` or `image_url`.*

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/moderate/image \
  -F "image_url=https://example.com/photo.jpg"
```

---

### 2. `POST /api/v1/moderate/video`
Moderates a video file (extracts frames and transcribes audio).

**Request (Multipart Form):**
- `file`: The video file to upload (`.mp4`, `.webm`, `.mov`).
- `caption`: (Optional) Contextual text.

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/moderate/video \
  -F "file=@/path/to/video.mp4"
```

---

### 3. `POST /api/v1/moderate/text`
Moderates raw text content.

**Request (JSON):**
```json
{
  "text": "The content to moderate"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/moderate/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

---

### 4. `POST /api/v1/moderate/pdf` & `POST /api/v1/moderate/docx`
Moderates documents (extracts text, metadata, embedded images, and links).

**Request (Multipart Form):**
- `file`: (Optional) The document file to upload.
- `document_url`: (Optional) A public HTTPS URL to the document.

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/moderate/pdf \
  -F "file=@/path/to/report.pdf"
```

---

### 5. `GET /api/v1/health`
Checks if the application is running.

**Response:**
```json
{
  "status": "ok",
  "service": "Aegis Moderation",
  "version": "1.0.0",
  "mode": "standalone"
}
```

---

### 6. `GET /api/v1/monitor/all`
Returns the full system monitoring state (used by the dashboard).

---

### 7. `GET /api/v1/metrics`
Prometheus metrics export. Includes request counts and latency histograms.

---

## Response Schema (Moderation)

All moderation endpoints return a normalized JSON report.

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
  "ocr_text": "Extracted text here",
  "content_type": "image",
  "document": { ... }, 
  "video": { ... }
}
```
