# cURL Examples

## Health

```bash
curl http://localhost:8000/api/v1/health
```

## Model Health

```bash
curl http://localhost:8000/api/v1/model-health
```

## Moderate Image URL

```bash
curl -X POST http://localhost:8000/api/v1/moderate \
  -H "Content-Type: application/json" \
  -d @examples/safe_text.json
```

## Upload Image

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@/path/to/image.jpg" \
  -F "caption=Optional context"
```
