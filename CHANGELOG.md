# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.1] — 2026-06-30

### Fixed
- Docker: Added missing `COPY --chown=aegis:aegis core/ ./core/` step to both `Dockerfile` and `Dockerfile.gpu`; `core.runtime` hardware backend was unreachable inside the container.
- `pyproject.toml`: Added `"core"` and `"core.runtime"` to setuptools `packages` so editable installs and wheels include the runtime backend.
- Removed debug artifacts (`scratch.py`, `monitor.json`, `full_logs.txt`) from tracking and added to `.gitignore`.
- Fixed UTF-8-SIG BOM and CP1252 mojibake in 18 Python source files.
- `backend/pipeline/surya_ocr.py`: Added `preprocess_for_ocr()` (CLAHE + Gaussian denoise + EXIF orient) for improved OCR accuracy on low-contrast documents.
- `backend/pipeline/video_moderation.py`: Dynamic frame sampling via `_probe_duration()` + `_compute_frame_fps()` replaces the fixed 1 fps strategy.

### Added
- **Phase 8 — Frontend completion**: `@media print` CSS block in `index.html` produces a clean A4 PDF when using browser "Print → Save as PDF". Hides input panel, adds report header, preserves score bars and category rows.
- **Phase 8 — Report generation**: "Copy JSON" clipboard button on the moderation report; "Export PDF" label replacing the generic "Print / PDF" label.
- **Phase 8 — Monitoring dashboard**: Decisions breakdown card (Accept / Review Required / Reject counts) and Recent Errors card added to `dashboard.html`.
- `.gitignore`: Entries for generated audit/diagnostic markdown reports.

---

## [1.0.0] — 2026-06-29

### Added
- Premium frontend redesign for `index.html` (glassmorphism, toast notifications, animations).
- New monitoring dashboard (`frontend/dashboard.html`) for real-time observability.
- CPU-default `Dockerfile` for broader compatibility (Windows/Mac/Linux out of the box).
- `docker-compose.gpu.yml` and `Dockerfile.gpu` for NVIDIA CUDA-accelerated deployments.
- Standalone FastAPI application with zero external service dependencies.
- REST API endpoints: `POST /api/v1/moderate/{image,video,text,pdf,docx}`.
- Health endpoints: `GET /api/v1/health`, `GET /api/v1/model-health`, `GET /api/v1/monitor/all`.
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
- Prometheus metrics at `GET /metrics`.
- Environment-variable driven size limits (`MAX_IMAGE_SIZE_MB`, `MAX_IMAGE_PIXELS`).
- GitHub Actions: tests, lint (ruff + black + mypy), Docker CI with health check smoke test.

### Changed
- Docker structure: Default build is now CPU-only. GPU build moved to `Dockerfile.gpu`.
- Frontend converted from React + Vite (requires Node.js build) to pure standalone HTML.
- `requirements.txt` replaced `fasttext` with `fasttext-wheel` for binary compatibility; added `httpx`.
- `pyproject.toml` test discovery widened to all `test_*.py` files.
- Lint workflow scoped to `backend/` for mypy to avoid stubs noise.

### Removed
- Supabase, PostgreSQL, SQLite, Redis, auth service, queue, and admin dashboard dependencies.
- Stub-only empty directories: `api/`, `classifier/`, `ocr/`, `rule_engine/`, `vision/`, `docker/`, `monitoring/`.
- `backend/visibility.py` — legacy post-visibility helper (not used by standalone pipeline).
- `runtime.txt` — Heroku deployment artifact.
- React / Node.js build dependency from production Dockerfile.
