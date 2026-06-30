# syntax=docker/dockerfile:1.7
# ── Aegis Moderation — CPU Dockerfile (default) ───────────────────────────
# No GPU required. Works on Windows (Docker Desktop + WSL2), Linux, macOS.
# Usage:  docker compose up --build
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CACHE_DIR=/root/.cache/pip \
    MODEL_WARMUP=false \
    XDG_CACHE_HOME=/home/aegis/.cache \
    HF_HOME=/home/aegis/.cache/huggingface \
    TORCH_HOME=/home/aegis/.cache/torch \
    YOLO_CONFIG_DIR=/home/aegis/.config/ultralytics

WORKDIR /app

ARG TORCH_CPU_WHEEL=https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl
ARG TORCHVISION_CPU_WHEEL=https://download.pytorch.org/whl/cpu/torchvision-0.22.1%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin aegis

COPY requirements.txt constraints.cpu.txt ./

# Step 1: Upgrade pip
RUN python -m pip install --upgrade pip wheel setuptools

# Step 2: Install CPU-only PyTorch (much smaller than CUDA variant)
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pip-wheels \
    curl -fL --retry 30 --retry-all-errors --retry-delay 5 --connect-timeout 30 \
    -C - -o /root/.cache/pip-wheels/torch-2.7.1+cpu-cp311-cp311-manylinux_2_28_x86_64.whl "$TORCH_CPU_WHEEL" \
    && curl -fL --retry 30 --retry-all-errors --retry-delay 5 --connect-timeout 30 \
    -C - -o /root/.cache/pip-wheels/torchvision-0.22.1+cpu-cp311-cp311-manylinux_2_28_x86_64.whl "$TORCHVISION_CPU_WHEEL" \
    && python -m pip install --default-timeout=1000 --retries=10 --resume-retries=20 \
    /root/.cache/pip-wheels/torch-2.7.1+cpu-cp311-cp311-manylinux_2_28_x86_64.whl \
    /root/.cache/pip-wheels/torchvision-0.22.1+cpu-cp311-cp311-manylinux_2_28_x86_64.whl

# Step 3: Install all remaining requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --default-timeout=1000 --retries=10 --resume-retries=20 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -c constraints.cpu.txt \
    -r requirements.txt

COPY --chown=aegis:aegis frontend/ ./frontend/
COPY --chown=aegis:aegis backend/ ./backend/
COPY --chown=aegis:aegis core/ ./core/
COPY --chown=aegis:aegis scripts/ ./scripts/

RUN mkdir -p /home/aegis/.cache/huggingface /home/aegis/.cache/torch \
    /home/aegis/.config/ultralytics /app/data /app/logs \
    && chown -R aegis:aegis /home/aegis /app

USER aegis
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
