"""Supabase persistence adapter for the MyItihas moderation service."""

from collections.abc import Callable
from functools import lru_cache
import logging
import os
import time
from typing import TypeVar
from urllib.error import URLError

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar("T")
DEFAULT_SUPABASE_RETRY_ATTEMPTS = 3
DEFAULT_SUPABASE_RETRY_BACKOFF_SECONDS = 0.5


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Create and cache a Supabase service-role client from environment values."""

    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not service_role_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be configured."
        )

    return create_client(supabase_url, service_role_key)


def execute_with_retries(
    operation: Callable[[], T],
    *,
    action: str,
    attempts: int | None = None,
    backoff_seconds: float | None = None,
) -> T:
    """Run a Supabase operation with short exponential backoff for transient failures."""

    max_attempts = attempts or int(
        os.getenv("SUPABASE_RETRY_ATTEMPTS", str(DEFAULT_SUPABASE_RETRY_ATTEMPTS))
    )
    base_backoff = backoff_seconds or float(
        os.getenv(
            "SUPABASE_RETRY_BACKOFF_SECONDS",
            str(DEFAULT_SUPABASE_RETRY_BACKOFF_SECONDS),
        )
    )

    last_error: Exception | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if not _is_retryable_exception(exc):
                break
            if attempt >= max_attempts:
                break
            delay = base_backoff * (2 ** (attempt - 1))
            logger.warning(
                "Supabase operation failed; retrying",
                extra={
                    "action": action,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "retry_delay_seconds": delay,
                },
            )
            time.sleep(delay)

    assert last_error is not None
    logger.error(
        "Supabase operation failed permanently",
        extra={"action": action, "max_attempts": max_attempts},
        exc_info=last_error,
    )
    raise last_error


def _is_retryable_exception(exc: Exception) -> bool:
    """Return True for transient transport/server errors."""

    if isinstance(exc, (ConnectionError, TimeoutError, OSError, URLError)):
        return True
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status_code, int):
        return status_code >= 500 or status_code == 429
    return False


def write_results(
    post_id: str,
    image_url: str,
    scores: dict[str, float],
    category: str,
    confidence: float,
    decision: str,
    reason: str,
    ocr_text: str,
    detected_objects: list | None = None,
    generated_caption: str | None = None,
    model_versions: dict | None = None,
    image_hash: str | None = None,
    # Phase 5 evidence fields
    qwen_description: str | None = None,
    qwen_confidence: float | None = None,
    uncertainty: float | None = None,
    similar_images: list | None = None,
    # Phase 6 multi-content-type fields
    content_type: str = "image",
    transcript: str | None = None,
    frame_scores: list | None = None,
    text_scores: dict | None = None,
    video_scores: dict | None = None,
) -> None:
    """Persist moderation scores, decision details, and post status updates."""

    import json as _json

    def _json_default(v: object) -> object:
        try:
            return float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return str(v)

    client = get_supabase_client()

    payload: dict = {
        "p_post_id": post_id,
        "p_adult_score": scores.get("adult_score", 0.0),
        "p_heritage_score": scores.get("heritage_score", 0.0),
        "p_child_safety_score": scores.get("child_safety_score", 0.0),
        "p_violence_self_harm_score": scores.get("violence_self_harm_score", 0.0),
        "p_content_quality_score": scores.get("content_quality_score", 0.0),
        "p_decision": decision,
        "p_reason": reason,
        "p_category_name": category,
        "p_category_confidence": confidence,
        "p_ocr_text": ocr_text,
    }

    # Evidence fields — passed as JSON strings; the RPC can ignore them if the
    # column doesn't exist yet (Supabase ignores unknown RPC params).
    if detected_objects is not None:
        payload["p_detected_objects"] = _json.dumps(detected_objects)
    if generated_caption:
        payload["p_generated_caption"] = generated_caption
    if model_versions:
        payload["p_model_versions"] = _json.dumps(model_versions)
    if image_hash:
        payload["p_image_hash"] = image_hash
    # Phase 5 evidence
    if qwen_description:
        payload["p_qwen_description"] = qwen_description[:500]
    if qwen_confidence is not None:
        payload["p_qwen_confidence"] = round(float(qwen_confidence), 4)
    if uncertainty is not None:
        payload["p_uncertainty"] = round(float(uncertainty), 4)
    if similar_images:
        payload["p_similar_images"] = _json.dumps(similar_images[:5])
    # Phase 6 multi-content-type fields
    payload["p_content_type"] = content_type
    if transcript:
        payload["p_transcript"] = transcript[:10000]
    if frame_scores:
        payload["p_frame_scores"] = _json.dumps(frame_scores[:300], default=_json_default)
    if text_scores:
        payload["p_text_scores"] = _json.dumps(text_scores, default=_json_default)
    if video_scores:
        payload["p_video_scores"] = _json.dumps(video_scores, default=_json_default)

    execute_with_retries(
        lambda: client.rpc("persist_moderation_result", payload).execute(),
        action="persist_moderation_result",
    )

    if decision == "UNDER_REVIEW":
        # Transition any active job (PENDING/PROCESSING) to UNDER_REVIEW so the
        # admin review queue can surface it. Falls back to insert for posts that
        # arrived through the API directly (no worker-created job exists yet).
        update_resp = execute_with_retries(
            lambda: client.table("moderation_jobs")
                .update({"status": "UNDER_REVIEW", "image_url": image_url})
                .eq("post_id", post_id)
                .in_("status", ["PENDING", "PROCESSING", "UNDER_REVIEW"])
                .execute(),
            action="update_moderation_job_under_review",
        )
        if not getattr(update_resp, "data", None):
            execute_with_retries(
                lambda: client.table("moderation_jobs").insert(
                    {
                        "post_id": post_id,
                        "image_url": image_url,
                        "status": "UNDER_REVIEW",
                    }
                ).execute(),
                action="insert_moderation_job_under_review",
            )

    if decision == "REJECTED" and scores.get("child_safety_score", 0.0) > 0.10:
        logger.critical(
            "Child safety moderation rejection",
            extra={"post_id": post_id, "decision": decision},
        )
