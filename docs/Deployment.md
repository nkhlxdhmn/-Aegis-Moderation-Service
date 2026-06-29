# Deployment

## Docker Compose — GPU (Production)

Requires NVIDIA Container Toolkit on the host.

```bash
docker compose up --build
```

The image is built from the NVIDIA CUDA 12.4 runtime base. Models are downloaded lazily on first request and cached in the `hf_cache` Docker volume.

Dashboard: **http://localhost:8000**
API docs: **http://localhost:8000/api/docs**

### GPU setup (host)

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Docker Compose — CPU (Development)

No GPU required. Models run on CPU (slower).

```bash
docker compose -f docker-compose.dev.yml up --build
```

---

## Local Python

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Pre-download models (optional — also downloads on first request)
python scripts/setup_models.py

uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## Health Checks

The application exposes a health endpoint used by Docker's healthcheck and load balancers:

```bash
curl http://localhost:8000/api/v1/health
# {"status":"ok","service":"Aegis Moderation","version":"1.0.0","mode":"standalone"}
```

Model readiness (lazy-load status):

```bash
curl http://localhost:8000/api/v1/model-health
```

---

## Model Warmup

To avoid cold-start latency on the first request, pre-load all models at startup:

```bash
# Via environment variable
MODEL_WARMUP=true uvicorn backend.main:app ...

# Via script
python scripts/warmup.py

# Via validation checklist
python backend/validate_models.py
```

---

## Volumes

| Volume | Contents |
|---|---|
| `hf_cache` | HuggingFace model weights (~10–50 GB depending on enabled models) |
| `app_data` | FAISS embedding index, hash dedup cache, calibration data |

---

## Prometheus Metrics

Metrics are available at `GET /metrics` or `GET /api/v1/metrics` in the Prometheus text format. Scrape config:

```yaml
scrape_configs:
  - job_name: aegis
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: /metrics
```

---

## Production Checklist

- [ ] GPU with ≥8 GB VRAM for fast inference (16 GB recommended for full pipeline)
- [ ] NVIDIA Container Toolkit installed on host
- [ ] `hf_cache` volume preserved across container restarts
- [ ] `MODEL_WARMUP=true` set so first request is not slow
- [ ] Health check endpoint monitored by your load balancer
- [ ] Firewall: expose port 8000 only to trusted networks (no auth built-in)
- [ ] Log aggregation pointed at container stdout (structured JSON)
