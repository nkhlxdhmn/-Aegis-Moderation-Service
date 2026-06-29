# API

Base URL: `http://localhost:8000`

## `GET /api/v1/health`

Returns service status.

```json
{
  "status": "ok",
  "service": "Aegis Moderation",
  "version": "1.0.0",
  "mode": "standalone"
}
```

## `GET /api/v1/model-health`

Reports model loading mode. Models are lazy-loaded or downloaded by their libraries on first analysis.

## `POST /api/v1/analyze`

Multipart form endpoint used by the browser UI.

Fields:

- `file`: optional image upload.
- `image_url`: optional public HTTPS image URL.
- `caption`: optional context string.

Provide exactly one of `file` or `image_url`.

## `POST /api/v1/moderate`

JSON endpoint for image URL analysis.

```json
{
  "image_url": "https://example.com/image.jpg",
  "caption": "optional context"
}
```

Response:

```json
{
  "overall_score": 83.2,
  "risk_level": "HIGH RISK",
  "decision": "Review Required",
  "categories": {
    "weapons": 74.1,
    "adult_content": 0.4
  },
  "objects": ["knife", "person"],
  "ocr_text": "...",
  "recommendation": "Human Review"
}
```

## OpenAPI

FastAPI serves OpenAPI at `/api/openapi.json` and interactive docs at `/api/docs`.
