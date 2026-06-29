"""Aegis Moderation standalone FastAPI application.

The app serves a browser UI and versioned JSON endpoints for image moderation. It does
not require a database, queue, Redis, or API key; every request is processed in-process
and returned as a report.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl
from starlette.concurrency import run_in_threadpool

from backend.image_io import ImageInputError, download_image, write_upload
from backend.reports import build_report
from pipeline.safety_flags import analyze_image

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
except Exception:  # pragma: no cover - metrics are optional at runtime
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    Counter = Histogram = None
    generate_latest = None

APP_NAME = "Aegis Moderation"
APP_VERSION = "1.0.0"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("aegis")

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="Standalone AI-powered image moderation platform with a browser dashboard.",
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

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if Counter and Histogram:
    REQUEST_COUNT = Counter("aegis_requests_total", "Total API requests", ["endpoint", "status"])
    ANALYSIS_LATENCY = Histogram("aegis_analysis_seconds", "Image analysis latency in seconds")
else:  # pragma: no cover
    REQUEST_COUNT = None
    ANALYSIS_LATENCY = None


class AnalyzeUrlRequest(BaseModel):
    """JSON request body for URL-based moderation."""

    image_url: HttpUrl = Field(..., description="Public HTTPS image URL to analyze.")
    caption: str | None = Field(default=None, max_length=2_000)


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
    return FileResponse(index_path)


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return process health for Docker and browser checks."""

    return HealthResponse(status="ok", service=APP_NAME, version=APP_VERSION, mode="standalone")


@app.get("/api/v1/model-health")
def model_health() -> dict[str, Any]:
    """Return model readiness metadata without forcing downloads."""

    return {
        "status": "ready",
        "mode": "lazy-load",
        "message": "Models are loaded or downloaded by the pipeline on first analysis.",
        "components": ["OCR", "Text Classifier", "Vision Models", "Rule Engine"],
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


async def _analyze_path(image_path: Path, caption: str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = await run_in_threadpool(analyze_image, str(image_path), caption)
        report = build_report(result)
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
            REQUEST_COUNT.labels(endpoint="analyze", status=locals().get("status", "error")).inc()
        image_path.unlink(missing_ok=True)


@app.post("/api/v1/analyze")
async def analyze_upload_or_url(
    file: Annotated[UploadFile | None, File()] = None,
    image_url: Annotated[str | None, Form()] = None,
    caption: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """Analyze an uploaded image or public HTTPS image URL."""

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

    report = await _analyze_path(image_path, caption)
    return JSONResponse(report)


@app.post("/api/v1/moderate")
async def moderate_url(request: AnalyzeUrlRequest) -> JSONResponse:
    """Analyze an image URL with a JSON request body."""

    try:
        image_path = await run_in_threadpool(download_image, str(request.image_url))
    except ImageInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    report = await _analyze_path(image_path, request.caption)
    return JSONResponse(report)


@app.get("/health", include_in_schema=False)
def legacy_health() -> HealthResponse:
    """Compatibility alias for simple container checks."""

    return health()
