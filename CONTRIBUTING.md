# Contributing to Aegis Moderation

Thank you for your interest in contributing to Aegis Moderation!

This project is a standalone, AI-powered multimodal content moderation platform designed for privacy, speed, and ease of deployment.

## Local Development

### Requirements
- Python 3.11+
- Make
- Docker (optional but recommended)

### Setup

```bash
git clone https://github.com/nkhlxdhmn/-Aegis-Moderation-Service.git
cd -Aegis-Moderation-Service
python -m venv .venv
# source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate           # Windows

make install
```

### Pre-warm Models (Optional)

Model weights are downloaded automatically on first use, but you can download them ahead of time:

```bash
python scripts/setup_models.py
```

### Running the Server

```bash
make run
# Opens at http://localhost:8000
```

## Docker Workflows

The repository uses two Docker configurations:

1. **CPU Default (Local testing/Dev)**
   ```bash
   docker compose up --build
   ```
2. **GPU Optimized (Production/NVIDIA)**
   ```bash
   docker compose -f docker-compose.gpu.yml up --build
   ```

## Guidelines

1. **Standalone Architecture**: Do not add dependencies on external services (Redis, databases, cloud APIs). The app must remain 100% self-contained.
2. **Type Hints**: All new Python code must be fully type-hinted.
3. **Tests**: Add unit tests for new features. Ensure `make test` passes.
4. **Code Style**: Run `make format` (Ruff + Black) before submitting PRs.
5. **Frontend**: The frontend (`index.html`, `dashboard.html`) is pure vanilla JS/CSS. Do not introduce npm, build steps, or frameworks like React/Vue.

## Submitting a Pull Request

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Run `make lint` and `make test`
5. Open a Pull Request against `main`
