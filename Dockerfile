# syntax=docker/dockerfile:1.7
FROM node:22-alpine AS frontend-build

WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/aegis/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/aegis/.cache/huggingface \
    YOLO_CONFIG_DIR=/home/aegis/.config/ultralytics

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    python3-pip \
    curl \
    ca-certificates \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && useradd --create-home --shell /usr/sbin/nologin aegis

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip wheel setuptools \
    && python -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install -r requirements.txt

COPY --chown=aegis:aegis --from=frontend-build /frontend/dist ./frontend/dist
COPY --chown=aegis:aegis backend ./backend

RUN mkdir -p /home/aegis/.cache /home/aegis/.config /app/data \
    && chown -R aegis:aegis /home/aegis /app

USER aegis
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
