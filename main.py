"""FastAPI entrypoint for the MyItihas moderation microservice.

This service implements the architecture document's moderation API boundary:
it authenticates trusted callers, runs the scoring/category pipeline, persists
the outcome to Supabase, and returns the moderation decision payload.
"""

from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID
import base64
import binascii
import ipaddress
import logging
import os
from pathlib import Path
import socket
import tempfile
from urllib.parse import urlparse
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
import warnings

from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request as FastAPIRequest, Response, status
from pydantic import BaseModel, ConfigDict, Field

import heartbeat_service
import model_warmup
from pipeline import gpu_scheduler
from pipeline import metrics as _pipeline_metrics
from pipeline.category_engine import get_top_category
from pipeline.circuit_breaker import all_statuses as all_breaker_statuses
from pipeline.decision_engine import decide_with_reason_code
from pipeline.safety_flags import ModerationPipelineResult, analyze_image
import queue_service
import security
from security import RequestContext
from supabase_client import execute_with_retries, get_supabase_client, write_results

load_dotenv()
logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGE_SIDE_PIXELS = 10_000
IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 20
ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
}
PIPELINE_ERROR_REASON = (
    "PIPELINE_ERROR: Critical moderation model failed; manual review required."
)

app = FastAPI(
    title="MyItihas Moderation Service",
    description="Content moderation pipeline for heritage and history posts.",
    version="0.1.0",
)


class ModerateRequest(BaseModel):
    """Request body for moderating a MyItihas post."""

    post_id: UUID
    image_url: str = Field(..., min_length=1)
    caption: str | None = None


class ModerateResponse(BaseModel):
    """Response body matching the moderation API response contract."""

    model_config = ConfigDict(protected_namespaces=())

    post_id: str
    decision: str
    reason: str
    category: str
    category_confidence: float
    scores: dict[str, float]
    category_scores: dict[str, float]
    ocr_text: str


class ReviewQueueItem(BaseModel):
    """Admin-facing summary of a moderation job awaiting review."""

    post_id: str
    image_url: str
    decision: str | None = None
    reason: str | None = None
    created_at: str | None = None
    adult_score: float
    child_safety_score: float
    violence_self_harm_score: float
    heritage_score: float
    scores: dict[str, float]


class ReviewQueueResponse(BaseModel):
    """Paginated response for the admin review queue."""

    items: list[ReviewQueueItem]
    limit: int
    offset: int
    count: int


class ReviewDetailResponse(BaseModel):
    """Admin-facing detail for one reviewed post."""

    post: dict[str, Any]
    moderation_result: dict[str, Any] | None
    ocr_text: str | None = None
    image_url: str | None = None
    decision: str | None = None
    reason: str | None = None


class AdminReviewActionRequest(BaseModel):
    """Optional metadata recorded with an admin review action."""

    reason: str | None = None


class AdminReviewActionResponse(BaseModel):
    """Response after an admin approves or rejects a reviewed post."""

    post_id: str
    admin_action: str
    moderation_status: str
    job_status: str


class ModerationMetricsResponse(BaseModel):
    """Admin moderation metrics snapshot."""

    pending: int
    processing: int
    under_review: int
    failed: int
    approved_today: int
    rejected_today: int


class QueueHealthResponse(BaseModel):
    """Admin queue health and worker status snapshot."""

    queue_depth: dict[str, int]
    oldest_pending_job: str | None
    oldest_pending_age_seconds: int | None
    workers: list[dict[str, Any]]


class HealthResponse(BaseModel):
    """Service readiness response."""

    status: str
    queue: str
    db: str
    models: str


class ModelHealthResponse(BaseModel):
    """Per-model state + GPU VRAM snapshot."""

    model_config = ConfigDict(protected_namespaces=())

    model_states: dict[str, str]
    circuit_breakers: list[dict]
    gpu_vram: list[dict]


class ImageInputError(RuntimeError):
    """Raised when the moderation image cannot be safely processed."""


def _allowed_image_hosts() -> set[str]:
    """Return the lowercase host allowlist for moderation image downloads."""

    configured = os.getenv("MODERATION_ALLOWED_IMAGE_HOSTS")
    if configured:
        return {
            host.strip().lower()
            for host in configured.split(",")
            if host.strip()
        }

    supabase_url = os.getenv("SUPABASE_URL")
    if not supabase_url:
        return set()

    hostname = urlparse(supabase_url).hostname
    return {hostname.lower()} if hostname else set()


def _resolve_public_host(hostname: str) -> None:
    """Reject hosts that resolve to non-public addresses."""

    try:
        address_info = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ImageInputError("Image URL host could not be resolved.") from exc

    if not address_info:
        raise ImageInputError("Image URL host could not be resolved.")

    for item in address_info:
        address = item[4][0]
        try:
            ip_address = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ImageInputError("Image URL resolved to an invalid address.") from exc

        if not ip_address.is_global:
            raise ImageInputError("Image URL host resolves to a non-public address.")


def _validate_image_url(image_url: str) -> None:
    """Validate a moderation image URL before network access."""

    parsed = urlparse(image_url)
    hostname = parsed.hostname.lower() if parsed.hostname else ""
    if parsed.scheme.lower() != "https":
        raise ImageInputError("Image URL must use HTTPS.")
    if not hostname:
        raise ImageInputError("Image URL host is required.")
    if parsed.username or parsed.password:
        raise ImageInputError("Image URL must not include credentials.")

    allowed_hosts = _allowed_image_hosts()
    if not allowed_hosts:
        raise ImageInputError("No allowed moderation image hosts are configured.")
    if hostname not in allowed_hosts:
        raise ImageInputError("Image URL host is not allowed.")

    _resolve_public_host(hostname)


class _ValidatingRedirectHandler(HTTPRedirectHandler):
    """Validate every redirect target before following it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _validate_image_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_pipeline_error(reason: str) -> ModerationPipelineResult:
    """Build an under-review compatible pipeline result for input failures."""

    return ModerationPipelineResult(
        scores={
            "adult_score": 0.0,
            "heritage_score": 0.0,
            "child_safety_score": 0.0,
            "violence_self_harm_score": 0.0,
            "content_quality_score": 0.0,
        },
        category_scores={},
        ocr_text="",
        pipeline_error=True,
        error_reason=reason,
    )


def _save_data_uri(image_url: str) -> str:
    """Decode a data:image/...;base64,... URI to a temp file and return its path."""

    # Strip the "data:" prefix, then split header from payload.
    try:
        header, encoded = image_url[5:].split(",", 1)
    except ValueError:
        raise ImageInputError("Malformed data URI: missing comma separator.")

    mime = header.split(";")[0].lower()
    if mime not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise ImageInputError(f"Data URI MIME type '{mime}' is not a supported image type.")

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageInputError("Data URI payload is not valid base64.") from exc

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise ImageInputError(f"Image exceeds {MAX_IMAGE_SIZE_MB} MB limit.")
    if len(image_bytes) == 0:
        raise ImageInputError("Image file is empty.")

    ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
           "image/gif": ".gif", "image/bmp": ".bmp"}.get(mime, ".img")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp.write(image_bytes)
        tmp.close()
    except OSError as exc:
        Path(tmp.name).unlink(missing_ok=True)
        raise ImageInputError("Failed to write data URI to temp file.") from exc

    logger.info("Data URI decoded to temp file")
    return tmp.name


def _download_image(image_url: str) -> str:
    """Download an image URL to a temporary file with a strict size limit.

    Also accepts data:image/...;base64,... URIs for testing via Swagger.
    """

    if image_url.startswith("data:"):
        return _save_data_uri(image_url)

    logger.info("Image download started")
    _validate_image_url(image_url)
    request = Request(image_url, headers={"User-Agent": "MyItihasModeration/1.0"})
    suffix = Path(image_url.split("?", 1)[0]).suffix or ".img"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    downloaded = 0

    try:
        with temp_file:
            opener = build_opener(_ValidatingRedirectHandler())
            with opener.open(request, timeout=IMAGE_DOWNLOAD_TIMEOUT_SECONDS) as response:
                content_type = response.headers.get_content_type()
                if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
                    raise ImageInputError("Image URL did not return a supported image type.")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > MAX_IMAGE_SIZE_BYTES:
                        raise ImageInputError(
                            f"Image exceeds {MAX_IMAGE_SIZE_MB} MB limit."
                        )
                    temp_file.write(chunk)
    except (OSError, URLError, ImageInputError) as exc:
        Path(temp_file.name).unlink(missing_ok=True)
        logger.exception("Image download failed")
        if isinstance(exc, ImageInputError):
            raise
        raise ImageInputError("Image could not be downloaded.") from exc

    if downloaded == 0:
        Path(temp_file.name).unlink(missing_ok=True)
        raise ImageInputError("Image file is empty.")

    logger.info("Image download completed")
    return temp_file.name


def _validate_image_file(image_path: str) -> None:
    """Reject empty, oversized, corrupted, or non-image files before inference."""

    logger.info("Image validation started")
    path = Path(image_path)
    size = path.stat().st_size
    if size == 0:
        raise ImageInputError("Image file is empty.")
    if size > MAX_IMAGE_SIZE_BYTES:
        raise ImageInputError(f"Image exceeds {MAX_IMAGE_SIZE_MB} MB limit.")

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        logger.exception("Pillow is required for image validation")
        raise ImageInputError("Image validation dependency is unavailable.") from exc

    try:
        # First open: read dimensions only.
        # Do NOT mutate the process-global Image.MAX_IMAGE_PIXELS — that is not
        # thread-safe under uvicorn's sync-route thread pool.  Instead we check
        # pixel counts explicitly and rely on DecompressionBombWarning (which
        # fires at PIL's own default limit) as a secondary backstop.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ImageInputError("Image dimensions are invalid.")
                if width > MAX_IMAGE_SIDE_PIXELS or height > MAX_IMAGE_SIDE_PIXELS:
                    raise ImageInputError(
                        f"Image dimensions exceed {MAX_IMAGE_SIDE_PIXELS}px side limit."
                    )
                if width * height > MAX_IMAGE_PIXELS:
                    raise ImageInputError(
                        f"Image exceeds {MAX_IMAGE_PIXELS} pixel limit."
                    )

        # Second open: verify() must be called on a freshly opened image before
        # any pixel data has been decoded — Pillow's docs require this.
        with Image.open(path) as verify_image:
            verify_image.verify()
    except ImageInputError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        logger.exception("Image decompression-bomb protection rejected file")
        raise ImageInputError("Image dimensions exceed safe processing limits.") from exc
    except (UnidentifiedImageError, OSError) as exc:
        logger.exception("Image validation failed")
        raise ImageInputError("Image is corrupted or is not a supported image file.") from exc

    logger.info("Image validation completed")


def _with_reason_code(code: str, reason: str) -> str:
    """Prefix a human-readable reason with a stable reason code."""

    return f"{code}: {reason}"


def _require_api_key(x_api_key: str | None) -> None:
    """Require the configured shared API key for trusted service callers."""

    security.require_api_key(x_api_key)


def _first_row(response: Any) -> dict[str, Any] | None:
    """Return the first row from a Supabase response."""

    data = getattr(response, "data", None)
    if not data:
        return None
    return data[0]


def _score_payload(row: dict[str, Any] | None) -> dict[str, float]:
    """Extract moderation score columns into a stable nested response shape."""

    row = row or {}
    return {
        "adult_score": float(row.get("adult_score") or 0.0),
        "heritage_score": float(row.get("heritage_score") or 0.0),
        "child_safety_score": float(row.get("child_safety_score") or 0.0),
        "violence_self_harm_score": float(
            row.get("violence_self_harm_score") or 0.0
        ),
        "content_quality_score": float(row.get("content_quality_score") or 0.0),
    }


def _latest_moderation_results_by_post_id(
    post_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch latest moderation result rows for a page of post IDs."""

    if not post_ids:
        return {}

    client = get_supabase_client()
    response = (
        client.table("moderation_results")
        .select("*")
        .in_("post_id", post_ids)
        .order("created_at", desc=True)
        .execute()
    )

    latest_by_post_id: dict[str, dict[str, Any]] = {}
    for row in getattr(response, "data", None) or []:
        post_id = str(row.get("post_id"))
        if post_id not in latest_by_post_id:
            latest_by_post_id[post_id] = row
    return latest_by_post_id


def _get_latest_moderation_result(post_id: str) -> dict[str, Any] | None:
    """Fetch the latest moderation result row for one post."""

    client = get_supabase_client()
    response = (
        client.table("moderation_results")
        .select("*")
        .eq("post_id", post_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return _first_row(response)


def _get_latest_moderation_job(post_id: str) -> dict[str, Any] | None:
    """Fetch the latest moderation job row for one post."""

    client = get_supabase_client()
    response = (
        client.table("moderation_jobs")
        .select("*")
        .eq("post_id", post_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return _first_row(response)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


@app.on_event("startup")
def startup() -> None:
    """Warm heavy models when enabled by deployment configuration."""

    model_warmup.warmup_models_if_enabled()


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint."""

    return {"service": "MyItihas Moderation Service", "status": "running"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service, queue, DB, and model readiness."""

    db_status = "ok"
    queue_status = "ok"
    try:
        queue_service.get_queue_depth_by_status()
    except Exception:
        logger.warning("Queue metrics unavailable during health check")
        queue_status = "error"

    models = model_warmup.model_status()
    service_status = "healthy" if models != "error" else "unhealthy"
    return HealthResponse(
        status=service_status,
        queue=queue_status,
        db=db_status,
        models=models,
    )


@app.get("/model-health", response_model=ModelHealthResponse)
def model_health() -> ModelHealthResponse:
    """Return per-model load state, circuit breaker status, and GPU VRAM usage."""

    model_states = model_warmup.model_status_detail()
    gpu_vram = gpu_scheduler.status_dicts()

    return ModelHealthResponse(
        model_states=model_states,
        circuit_breakers=all_breaker_statuses(),
        gpu_vram=gpu_vram,
    )


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics scrape endpoint (compatible with standard scrapers)."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return Response(
            content="# prometheus_client not installed\n",
            media_type="text/plain; version=0.0.4",
        )


@app.get("/admin/review-queue", response_model=ReviewQueueResponse)
def get_review_queue(
    admin: Annotated[RequestContext, Depends(security.require_admin)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ReviewQueueResponse:
    """Return under-review moderation jobs for admin triage."""

    try:
        client = get_supabase_client()
        response = (
            client.table("moderation_jobs")
            .select("post_id,image_url,created_at")
            .eq("status", "UNDER_REVIEW")
            .order("created_at", desc=False)
            .range(offset, offset + limit - 1)
            .execute()
        )
        jobs = getattr(response, "data", None) or []
        result_by_post_id = _latest_moderation_results_by_post_id(
            [str(job["post_id"]) for job in jobs]
        )
    except Exception as exc:
        logger.exception("Failed to fetch admin review queue")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch review queue.",
        ) from exc

    items = []
    for job in jobs:
        post_id = str(job["post_id"])
        result = result_by_post_id.get(post_id)
        scores = _score_payload(result)
        items.append(
            ReviewQueueItem(
                post_id=post_id,
                image_url=str(job["image_url"]),
                decision=result.get("decision") if result else None,
                reason=result.get("reason") if result else None,
                created_at=job.get("created_at"),
                adult_score=scores["adult_score"],
                child_safety_score=scores["child_safety_score"],
                violence_self_harm_score=scores["violence_self_harm_score"],
                heritage_score=scores["heritage_score"],
                scores=scores,
            )
        )

    return ReviewQueueResponse(
        items=items,
        limit=limit,
        offset=offset,
        count=len(items),
    )


@app.get("/admin/review/{post_id}", response_model=ReviewDetailResponse)
def get_review_detail(
    post_id: UUID,
    admin: Annotated[RequestContext, Depends(security.require_admin)] = None,
) -> ReviewDetailResponse:
    """Return post and moderation details for one admin review item."""

    post_id_str = str(post_id)
    try:
        client = get_supabase_client()
        post_response = (
            client.table("posts")
            .select("*")
            .eq("id", post_id_str)
            .limit(1)
            .execute()
        )
        post = _first_row(post_response)
        if post is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Post was not found.",
            )

        moderation_result = _get_latest_moderation_result(post_id_str)
        moderation_job = _get_latest_moderation_job(post_id_str)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to fetch admin review detail for post %s", post_id_str)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch review detail.",
        ) from exc

    return ReviewDetailResponse(
        post=post,
        moderation_result=moderation_result,
        ocr_text=post.get("ocr_text"),
        image_url=moderation_job.get("image_url") if moderation_job else None,
        decision=moderation_result.get("decision") if moderation_result else None,
        reason=moderation_result.get("reason") if moderation_result else None,
    )


@app.post(
    "/admin/review/{post_id}/approve",
    response_model=AdminReviewActionResponse,
)
def approve_review_item(
    http_request: FastAPIRequest,
    post_id: UUID,
    request: Annotated[AdminReviewActionRequest | None, Body()] = None,
    admin: Annotated[RequestContext, Depends(security.require_admin)] = None,
) -> AdminReviewActionResponse:
    """Approve a post that was waiting for admin moderation review."""

    return _apply_admin_review_action(
        str(post_id),
        "APPROVE",
        "ADMIN_APPROVED",
        request,
        admin,
        http_request,
    )


@app.post(
    "/admin/review/{post_id}/reject",
    response_model=AdminReviewActionResponse,
)
def reject_review_item(
    http_request: FastAPIRequest,
    post_id: UUID,
    request: Annotated[AdminReviewActionRequest | None, Body()] = None,
    admin: Annotated[RequestContext, Depends(security.require_admin)] = None,
) -> AdminReviewActionResponse:
    """Reject a post that was waiting for admin moderation review."""

    return _apply_admin_review_action(
        str(post_id),
        "REJECT",
        "ADMIN_REJECTED",
        request,
        admin,
        http_request,
    )


def _apply_admin_review_action(
    post_id: str,
    action: str,
    moderation_status: str,
    request: AdminReviewActionRequest | None,
    admin: RequestContext,
    http_request: FastAPIRequest,
) -> AdminReviewActionResponse:
    """Persist an admin review action and close matching moderation jobs."""

    request = request or AdminReviewActionRequest()
    rpc_name = "approve_review" if action == "APPROVE" else "reject_review"

    try:
        client = get_supabase_client()
        rpc_params = {
            "p_post_id": post_id,
            "p_reason": request.reason,
            "p_admin_id": admin.admin_id,
            "p_ip_address": security.client_ip(http_request),
            "p_request_id": admin.request_id,
        }
        execute_with_retries(
            lambda: client.rpc(rpc_name, rpc_params).execute(),
            action=rpc_name,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to apply admin review action for post %s", post_id)
        # The approve_review / reject_review RPCs RAISE with message containing
        # "No UNDER_REVIEW moderation job" when the post has already been reviewed
        # (idempotent-retry case) or genuinely has no pending review job.  We map
        # that to 404 so the caller knows the post is no longer actionable rather
        # than receiving a misleading 502.
        exc_text = str(exc).lower()
        if "no under_review moderation job" in exc_text or "post" in exc_text and "not found" in exc_text:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No pending review job was found for this post.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to apply admin review action.",
        ) from exc

    return AdminReviewActionResponse(
        post_id=post_id,
        admin_action=action,
        moderation_status=moderation_status,
        job_status="COMPLETED",
    )


@app.get("/admin/moderation/metrics", response_model=ModerationMetricsResponse)
def get_moderation_metrics(
    admin: Annotated[RequestContext, Depends(security.require_admin)] = None,
) -> ModerationMetricsResponse:
    """Return moderation queue and daily decision counts."""

    try:
        counts = queue_service.get_queue_depth_by_status()
        decision_counts = queue_service.get_decision_counts_today()
    except Exception as exc:
        logger.exception("Failed to fetch moderation metrics")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch moderation metrics.",
        ) from exc

    return ModerationMetricsResponse(
        pending=counts.get("PENDING", 0),
        processing=counts.get("PROCESSING", 0),
        under_review=counts.get("UNDER_REVIEW", 0),
        failed=counts.get("FAILED", 0),
        approved_today=decision_counts.get("approved_today", 0),
        rejected_today=decision_counts.get("rejected_today", 0),
    )


@app.get("/admin/moderation/queue-health", response_model=QueueHealthResponse)
def get_queue_health(
    admin: Annotated[RequestContext, Depends(security.require_admin)] = None,
) -> QueueHealthResponse:
    """Return queue health details and worker online status."""

    try:
        depth = queue_service.get_queue_depth_by_status()
        oldest_pending = queue_service.get_oldest_pending_job_created_at()
        parsed_oldest = _parse_timestamp(oldest_pending)
        oldest_age = (
            int((datetime.now(UTC) - parsed_oldest).total_seconds())
            if parsed_oldest
            else None
        )
        workers = heartbeat_service.get_worker_statuses()
    except Exception as exc:
        logger.exception("Failed to fetch moderation queue health")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch moderation queue health.",
        ) from exc

    return QueueHealthResponse(
        queue_depth=depth,
        oldest_pending_job=oldest_pending,
        oldest_pending_age_seconds=oldest_age,
        workers=workers,
    )


@app.post("/moderate", response_model=ModerateResponse)
def moderate(
    http_request: FastAPIRequest,
    request: ModerateRequest,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> ModerateResponse:
    """Moderate a post image and caption, then persist the final decision."""

    _require_api_key(x_api_key)
    security.check_rate_limit(http_request, "moderate", 60)
    request_id = security.request_id()

    post_id = str(request.post_id)
    temp_image_path: str | None = None
    with _pipeline_metrics.moderation_duration_seconds.time():
        try:
            temp_image_path = _download_image(request.image_url)
            _validate_image_file(temp_image_path)
            pipeline_result = analyze_image(temp_image_path, request.caption)
        except ImageInputError as exc:
            logger.exception("Image input rejected for moderation")
            pipeline_result = _default_pipeline_error(str(exc))
        finally:
            if temp_image_path is not None:
                Path(temp_image_path).unlink(missing_ok=True)

    scores = pipeline_result.scores
    category_scores = pipeline_result.category_scores
    ocr_text = pipeline_result.ocr_text
    category, category_confidence = get_top_category(category_scores)

    if pipeline_result.pipeline_error:
        decision = "UNDER_REVIEW"
        reason = PIPELINE_ERROR_REASON
        if pipeline_result.error_reason:
            reason = f"{reason} Detail: {pipeline_result.error_reason}"
    else:
        decision, reason_code, human_reason = decide_with_reason_code(scores)
        reason = _with_reason_code(reason_code, human_reason)

    _pipeline_metrics.moderation_requests_total.labels(decision=decision).inc()

    logger.info(
        "Moderation decision:\n"
        "post_id=%s\n"
        "decision=%s\n"
        "adult_score=%.4f\n"
        "heritage_score=%.4f\n"
        "content_quality_score=%.4f\n"
        "category_name=%s",
        post_id,
        decision,
        scores.get("adult_score", 0.0),
        scores.get("heritage_score", 0.0),
        scores.get("content_quality_score", 0.0),
        category,
        extra={
            "request_id": request_id,
            "post_id": post_id,
            "decision": decision,
        },
    )

    try:
        write_results(
            post_id=post_id,
            image_url=request.image_url,
            scores=scores,
            category=category,
            confidence=category_confidence,
            decision=decision,
            reason=reason,
            ocr_text=ocr_text,
            detected_objects=getattr(pipeline_result, "detected_objects", []),
            generated_caption=getattr(pipeline_result, "generated_caption", ""),
            model_versions=getattr(pipeline_result, "model_versions", {}),
            image_hash=getattr(pipeline_result, "image_hash", None),
        )
    except Exception as exc:
        logger.exception("Failed to persist moderation result for post %s", post_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Moderation decision was created but could not be saved.",
        ) from exc

    return ModerateResponse(
        post_id=post_id,
        decision=decision,
        reason=reason,
        category=category,
        category_confidence=category_confidence,
        scores=scores,
        category_scores=category_scores,
        ocr_text=ocr_text,
    )
