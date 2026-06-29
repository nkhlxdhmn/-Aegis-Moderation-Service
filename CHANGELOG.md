# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-06-29

### Added
- Standalone FastAPI application with zero external service dependencies.
- Browser dashboard (`frontend/index.html`) — pure HTML/CSS/JS, no build step required.
- Dark mode toggle with `prefers-color-scheme` detection and localStorage persistence.
- REST API endpoints: `POST /api/v1/moderate/{image,video,text,pdf,docx}`.
- Health endpoints: `GET /api/v1/health`, `GET /api/v1/model-health`, `GET /api/v1/metrics`.
- 11-stage image moderation pipeline: NSFW, YOLO11x, SigLIP2, hybrid OCR, BLIP captioning, Detoxify, PII detection, QR detection, rule engine, uncertainty estimation, FAISS embedding cache.
- Text moderation pipeline: Detoxify multilingual, FastText language ID, rule-based classifiers.
- Video moderation pipeline: ffmpeg frame extraction, per-frame image pipeline, Whisper ASR.
- PDF and DOCX document moderation with link analysis, phishing detection, and PII scanning.
- SSRF protection for all URL-based inputs (DNS resolution, RFC-1918 block).
- Zip bomb protection for DOCX uploads.
- Security response headers middleware (`X-Content-Type-Options`, `X-Frame-Options`).
- `scripts/setup_models.py` — pre-download all model weights.
- `scripts/warmup.py` — pre-warm models before accepting traffic.
- `scripts/benchmark.py` — latency benchmarking for OCR and end-to-end pipeline.
- `backend/validate_models.py` — pre-flight model validation checklist.
- Prometheus metrics at `GET /metrics` and `GET /api/v1/metrics`.
- Environment-variable driven size limits (`MAX_IMAGE_SIZE_MB`, `MAX_IMAGE_PIXELS`).
- GitHub Actions: tests, lint (ruff + black + mypy), Docker CI with health check smoke test.
- Dependabot config for GitHub Actions and pip.
- Issue templates and PR template.

### Changed
- Frontend converted from React + Vite (requires Node.js build) to pure standalone HTML — no npm, no build step.
- Dockerfile simplified: Node.js build stage removed; `frontend/index.html` copied directly.
- `requirements.txt` replaced `fasttext` with `fasttext-wheel` for binary compatibility; added `httpx`.
- `pyproject.toml` test discovery widened to all `test_*.py` files.
- Lint workflow scoped to `backend/` for mypy to avoid stubs noise.
- Docker CI switched to dev image (CPU, no NVIDIA driver required in GitHub runners).

### Removed
- Supabase, PostgreSQL, SQLite, Redis, auth service, queue, and admin dashboard dependencies.
- Stub-only empty directories: `api/`, `classifier/`, `ocr/`, `rule_engine/`, `vision/`, `docker/`, `monitoring/`.
- `backend/visibility.py` — legacy post-visibility helper (not used by standalone pipeline).
- `runtime.txt` — Heroku deployment artifact.
- `training/` stub — replaced by docs note about fine-tuning.
- Committed `__pycache__/`, `.mypy_cache/`, `.pytest_cache/`, `frontend/dist/` artefacts.
- React / Node.js build dependency from production Dockerfile.
