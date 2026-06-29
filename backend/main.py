"""Aegis Moderation standalone FastAPI application."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, HttpUrl
from starlette.concurrency import run_in_threadpool

from backend.documents import (
    DocumentInputError,
    download_document,
    moderate_docx,
    moderate_pdf,
    write_document_upload,
)
from backend.image_io import ImageInputError, download_image, write_upload
from backend.pipeline.safety_flags import analyze_image
from backend.pipeline.text_moderation import moderate_text
from backend.pipeline.video_moderation import moderate_video
from backend.reports import build_report

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
except Exception:  # pragma: no cover - metrics are optional at runtime
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    Counter = Histogram = None
    generate_latest = None

APP_NAME = "Aegis Moderation"
APP_VERSION = "1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("aegis")

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Standalone AI-powered multimodal content moderation platform.",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Security headers injected on every response.
@app.middleware("http")
async def _security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

if Counter and Histogram:
    REQUEST_COUNT = Counter("aegis_requests_total", "Total API requests", ["endpoint", "status"])
    ANALYSIS_LATENCY = Histogram("aegis_analysis_seconds", "Content analysis latency in seconds")
else:  # pragma: no cover
    REQUEST_COUNT = None
    ANALYSIS_LATENCY = None


class AnalyzeUrlRequest(BaseModel):
    """JSON request body for URL-based image moderation."""

    image_url: HttpUrl = Field(..., description="Public HTTPS image URL to analyze.")
    caption: str | None = Field(default=None, max_length=2_000)


class TextModerationRequest(BaseModel):
    """JSON request body for text moderation."""

    text: str = Field(..., min_length=1, max_length=100_000)


class HealthResponse(BaseModel):
    """Service health response."""

    status: str
    service: str
    version: str
    mode: str


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the standalone browser dashboard."""

    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Frontend assets are missing.")
    return FileResponse(index_path, media_type="text/html")


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return process health for Docker and browser checks."""

    return HealthResponse(status="ok", service=APP_NAME, version=APP_VERSION, mode="standalone")


@app.get("/api/v1/model-health")
def model_health() -> dict[str, Any]:
    """Return per-model readiness status without forcing any downloads."""

    try:
        from backend.model_warmup import model_status, model_status_detail
        detail = model_status_detail()
        overall = model_status()
    except Exception:
        detail = {}
        overall = "not_loaded"
    return {
        "status": overall,
        "mode": "lazy-load",
        "message": "Models load on first request or when MODEL_WARMUP=true.",
        "models": detail,
        "components": ["OCR", "Text Classifier", "Vision Models", "Rule Engine", "Document Parser"],
    }


@app.get("/api/v1/metrics", include_in_schema=False)
def versioned_metrics() -> PlainTextResponse:
    """Expose Prometheus metrics at a versioned path."""

    return metrics()


@app.get("/metrics", include_in_schema=False)
def metrics() -> PlainTextResponse:
    """Expose Prometheus metrics at the conventional scrape path."""

    if generate_latest is None:
        return PlainTextResponse("# prometheus_client is not installed\n", media_type="text/plain")
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


async def _write_binary_upload(
    file: UploadFile,
    *,
    allowed_suffixes: set[str],
    allowed_content_types: set[str],
    max_bytes: int,
) -> Path:
    """Persist an uploaded media file with extension, MIME, and size validation."""

    suffix = Path(file.filename or "upload.bin").suffix.lower()
    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400, detail=f"Supported extensions: {', '.join(sorted(allowed_suffixes))}."
        )
    if file.content_type and file.content_type not in allowed_content_types:
        raise HTTPException(status_code=400, detail="Unsupported media type.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Upload is empty.")
    if len(contents) > max_bytes:
        raise HTTPException(status_code=400, detail="Upload exceeds the size limit.")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp.write(contents)
        temp.close()
        return Path(temp.name)
    except Exception:
        Path(temp.name).unlink(missing_ok=True)
        raise


async def _analyze_image_path(image_path: Path, caption: str | None = None) -> dict[str, Any]:
    """Run the image pipeline and normalize its public report."""

    started = time.perf_counter()
    try:
        result = await run_in_threadpool(analyze_image, str(image_path), caption)
        report = build_report(result)
        report["content_type"] = "image"
        status = "ok"
        return report
    except Exception as exc:  # pragma: no cover - depends on optional model stack
        logger.exception("Image analysis failed")
        status = "error"
        raise HTTPException(status_code=500, detail="Image analysis failed.") from exc
    finally:
        if ANALYSIS_LATENCY:
            ANALYSIS_LATENCY.observe(time.perf_counter() - started)
        if REQUEST_COUNT:
            REQUEST_COUNT.labels(endpoint="image", status=locals().get("status", "error")).inc()
        image_path.unlink(missing_ok=True)


async def _moderate_image_input(
    file: UploadFile | None,
    image_url: str | None,
    caption: str | None,
) -> JSONResponse:
    has_file = file is not None and bool(file.filename)
    has_url = bool(image_url and image_url.strip())
    if has_file == has_url:
        raise HTTPException(
            status_code=400, detail="Provide either an uploaded image or an image URL."
        )

    try:
        if has_file and file is not None:
            suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
            image_path = write_upload(await file.read(), file.content_type, suffix=suffix)
        else:
            image_path = await run_in_threadpool(download_image, image_url or "")
    except ImageInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(await _analyze_image_path(image_path, caption))


@app.post("/api/v1/moderate/image")
async def moderate_image_endpoint(
    file: Annotated[UploadFile | None, File()] = None,
    image_url: Annotated[str | None, Form()] = None,
    caption: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Analyze an uploaded image or public HTTPS image URL."""

    return await _moderate_image_input(file=file, image_url=image_url, caption=caption)


@app.post("/api/v1/moderate/video")
async def moderate_video_endpoint(
    file: Annotated[UploadFile, File()],
    caption: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Analyze an uploaded video file."""

    path = await _write_binary_upload(
        file,
        allowed_suffixes={".mp4", ".mov", ".mkv", ".webm"},
        allowed_content_types={
            "video/mp4",
            "video/quicktime",
            "video/x-matroska",
            "video/webm",
            "application/octet-stream",
        },
        max_bytes=250 * 1024 * 1024,
    )
    try:
        result = await run_in_threadpool(moderate_video, str(path), {"caption": caption or ""})
        report = build_report(result)
        report["content_type"] = "video"
        report["video"] = {
            "frame_count": getattr(result, "frame_count", 0),
            "unsafe_frame_count": getattr(result, "unsafe_frame_count", 0),
            "max_consecutive_unsafe": getattr(result, "max_consecutive_unsafe", 0),
            "transcript": getattr(result, "transcript", ""),
        }
        return JSONResponse(report)
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/v1/moderate/text")
async def moderate_text_endpoint(request: TextModerationRequest) -> JSONResponse:
    """Analyze submitted plain text."""

    result = await run_in_threadpool(moderate_text, request.text)
    report = build_report(result)
    report["content_type"] = "text"
    report["extracted_text_preview"] = request.text[:2000]
    return JSONResponse(report)


@app.post("/api/v1/moderate/pdf")
async def moderate_pdf_endpoint(
    file: Annotated[UploadFile | None, File()] = None,
    document_url: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Analyze an uploaded PDF or public HTTPS PDF URL."""

    return await _moderate_document_input(file, document_url, ".pdf", moderate_pdf, "pdf")


@app.post("/api/v1/moderate/docx")
async def moderate_docx_endpoint(
    file: Annotated[UploadFile | None, File()] = None,
    document_url: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Analyze an uploaded DOCX or public HTTPS DOCX URL."""

    return await _moderate_document_input(file, document_url, ".docx", moderate_docx, "docx")


async def _moderate_document_input(
    file: UploadFile | None,
    document_url: str | None,
    suffix: str,
    processor: Any,
    content_type: str,
) -> JSONResponse:
    has_file = file is not None and bool(file.filename)
    has_url = bool(document_url and document_url.strip())
    if has_file == has_url:
        raise HTTPException(
            status_code=400, detail=f"Provide either a {suffix.upper().lstrip('.')} upload or URL."
        )

    try:
        if has_file and file is not None:
            document = write_document_upload(
                await file.read(),
                file.filename or f"upload{suffix}",
                file.content_type,
                suffix,
            )
        else:
            document = await run_in_threadpool(download_document, document_url or "", suffix)
        report = await run_in_threadpool(processor, document)
        report["content_type"] = content_type
        return JSONResponse(report)
    except DocumentInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "document" in locals():
            document.path.unlink(missing_ok=True)


@app.post("/api/v1/analyze")
async def analyze_upload_or_url(
    file: Annotated[UploadFile | None, File()] = None,
    image_url: Annotated[str | None, Form()] = None,
    caption: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Compatibility alias for image analysis."""

    return await _moderate_image_input(file=file, image_url=image_url, caption=caption)


@app.post("/api/v1/moderate")
async def moderate_url(request: AnalyzeUrlRequest) -> JSONResponse:
    """Compatibility JSON endpoint for image URL moderation."""

    try:
        image_path = await run_in_threadpool(download_image, str(request.image_url))
    except ImageInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(await _analyze_image_path(image_path, request.caption))


@app.get("/health", include_in_schema=False)
def legacy_health() -> HealthResponse:
    """Compatibility alias for simple container checks."""

    return health()
