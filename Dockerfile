# syntax=docker/dockerfile:1.7
# ── Aegis Moderation — CPU Dockerfile (default) ───────────────────────────
# No GPU required. Works on Windows (Docker Desktop + WSL2), Linux, macOS.
# Usage:  docker compose up --build
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/aegis/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/aegis/.cache/huggingface \
    YOLO_CONFIG_DIR=/home/aegis/.config/ultralytics

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin aegis

COPY requirements.txt ./

# Step 1: Upgrade pip
RUN python -m pip install --upgrade pip wheel setuptools

# Step 2: Install CPU-only PyTorch (much smaller than CUDA variant)
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --default-timeout=300 --retries=5 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 3: Install all remaining requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --default-timeout=300 --retries=5 \
    -r requirements.txt

COPY --chown=aegis:aegis frontend/ ./frontend/
COPY --chown=aegis:aegis backend/ ./backend/
COPY --chown=aegis:aegis scripts/ ./scripts/

RUN mkdir -p /home/aegis/.cache/huggingface /home/aegis/.config /app/data \
    && chown -R aegis:aegis /home/aegis /app

USER aegis
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
