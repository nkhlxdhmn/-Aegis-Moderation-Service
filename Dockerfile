# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    python3-pip \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_PROGRESS_BAR=off \
    PIP_INSTALL_ATTEMPTS=4

RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install --upgrade pip wheel setuptools

# ── Stage 1: Core API runtime ─────────────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install \
    "fastapi>=0.115,<1.0" \
    "uvicorn[standard]>=0.30,<1.0" \
    "pydantic>=2.7,<3.0" \
    "python-dotenv>=1.0,<2.0" \
    "supabase>=2.9,<3.0" \
    "PyJWT>=2.8,<3.0" \
    "redis>=5.0,<6.0" \
    "prometheus_client>=0.20,<1.0"

# ── Stage 2: Image processing ─────────────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install \
    "pillow>=10.4,<12.0" \
    "numpy>=1.26,<3.0" \
    "opencv-python-headless>=4.10,<5.0" \
    "ImageHash>=4.3,<5.0"

# ── Stage 3: PyTorch with CUDA 12.4 ──────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    for attempt in $(seq 1 "$PIP_INSTALL_ATTEMPTS"); do \
        python3.11 -m pip install \
            torch==2.5.1+cu124 \
            torchvision==0.20.1+cu124 \
            --index-url https://download.pytorch.org/whl/cu124 && break; \
        if [ "$attempt" = "$PIP_INSTALL_ATTEMPTS" ]; then exit 1; fi; \
        sleep $((attempt * 10)); \
    done

# ── Stage 4: Hugging Face stack ───────────────────────────────────────────────
# transformers 4.49+ required for SigLIP2 and BLIP captioning
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install \
    "transformers>=4.49,<5.0" \
    "accelerate>=0.34,<1.0" \
    "sentencepiece>=0.2,<1.0" \
    "protobuf>=3.20,<5.0"

# ── Stage 5: Object detection YOLO11x ────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install "ultralytics>=8.3,<9.0"

# ── Stage 6: OpenNSFW2 + optional PII detection ───────────────────────────────
RUN --mount=type=cache,target=/root/.cache/pip \
    for attempt in $(seq 1 "$PIP_INSTALL_ATTEMPTS"); do \
        python3.11 -m pip install "opennsfw2>=0.10,<1.0" && exit 0; \
        if [ "$attempt" = "$PIP_INSTALL_ATTEMPTS" ]; then break; fi; \
        sleep $((attempt * 10)); \
    done; \
    true
RUN --mount=type=cache,target=/root/.cache/pip \
    for attempt in $(seq 1 "$PIP_INSTALL_ATTEMPTS"); do \
        python3.11 -m pip install "presidio-analyzer>=2.2,<3.0" && exit 0; \
        if [ "$attempt" = "$PIP_INSTALL_ATTEMPTS" ]; then break; fi; \
        sleep $((attempt * 10)); \
    done; \
    true

# ── Stage 7: FAISS embedding cache ───────────────────────────────────────────
# faiss-gpu pulls CUDA 12.9 wheels that conflict with torch==2.5.1+cu124.
# faiss-cpu is used; index operations run in microseconds and are not the bottleneck.
RUN --mount=type=cache,target=/root/.cache/pip \
    for attempt in $(seq 1 "$PIP_INSTALL_ATTEMPTS"); do \
        python3.11 -m pip install "faiss-cpu>=1.7,<2.0" && exit 0; \
        if [ "$attempt" = "$PIP_INSTALL_ATTEMPTS" ]; then break; fi; \
        sleep $((attempt * 10)); \
    done; \
    echo "WARNING: faiss not installed — embedding cache disabled"

# ── Stage 8: EasyOCR fallback + language/QR helpers ─────────────────────────
# easyocr     — Indic fallback OCR (pipeline/easyocr_engine.py)
# langdetect  — language identification (pipeline/language_detector.py)
# pyzbar      — QR / barcode detection (pipeline/qr_detector.py)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install \
        "easyocr>=1.7,<2.0" \
        "langdetect>=1.0,<2.0" \
        "pyzbar>=0.1,<1.0"

# ── Stage 9: Surya OCR (primary OCR engine) ──────────────────────────────────
# pipeline/surya_ocr.py loads this package at runtime; pipeline/ocr.py routes
# Surya → EasyOCR automatically when Surya is unavailable or produces no text.
# Uses --upgrade to ensure the latest compatible version is installed and to
# avoid stale cached layers from prior builds where surya may have failed.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m pip install --upgrade "surya-ocr>=0.4,<1.0" || \
    echo "WARNING: surya-ocr install failed — EasyOCR will serve as sole OCR engine"

# Verify surya is importable (fails silently — EasyOCR fallback handles absence)
RUN python3.11 -c "import surya; print('surya OK')" || \
    echo "WARNING: surya import check failed — EasyOCR fallback will be used"

# ── Stage 10: Text moderation packages ───────────────────────────────────────
# detoxify max release is 0.5.2 (PyPI top version is <1.0, no v1.x exists yet)
# fasttext  — language identification via lid.176.bin (pipeline/text_moderation.py)
RUN --mount=type=cache,target=/root/.cache/pip \
    for attempt in $(seq 1 "$PIP_INSTALL_ATTEMPTS"); do \
        python3.11 -m pip install \
            "detoxify>=0.3,<1.0" \
            "fasttext>=0.9,<1.0" && exit 0; \
        if [ "$attempt" = "$PIP_INSTALL_ATTEMPTS" ]; then break; fi; \
        sleep $((attempt * 10)); \
    done; \
    echo "WARNING: detoxify/fasttext install failed — text toxicity scoring disabled"

# ── Set cache directories before model pre-downloads ─────────────────────────
# Must be declared here so pre-download RUN steps see the correct paths.
# The docker-compose volume mounts hf_cache:/root/.cache/huggingface
# (not /app/.cache) so we keep the default HF paths for build-time downloads.
ENV HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    YOLO_CONFIG_DIR=/root/.config/ultralytics

# ── Pre-download YOLO11x weights (avoids first-request delay) ─────────────────
# Placed before COPY so code changes do not invalidate this layer.
RUN python3.11 -c "from ultralytics import YOLO; YOLO('yolo11x.pt')" || \
    echo "WARNING: YOLO11x pre-download skipped"

# ── Pre-download Whisper-large-v3 weights (avoids first-video-request delay) ───
# Placed before COPY for the same layer-cache reason.
# Runs on CPU — weights are device-agnostic; device assignment happens at
# runtime in video_moderation.py.
RUN python3.11 -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='openai/whisper-large-v3')" || \
    echo "WARNING: Whisper-large-v3 pre-download skipped"

# ── Copy application code ──────────────────────────────────────────────────────
COPY . .

# ── Runtime environment ───────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=8.9 \
    CUDA_VISIBLE_DEVICES=0,1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
