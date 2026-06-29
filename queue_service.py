"""Supabase-backed moderation job queue service.

This module only manages rows in public.moderation_jobs. It does not run
workers, perform moderation, or make moderation decisions.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from supabase_client import execute_with_retries, get_supabase_client

logger = logging.getLogger(__name__)

JOB_TABLE = "moderation_jobs"
CLAIM_JOB_RPC = "claim_next_moderation_job"
COMPLETE_JOB_RPC = "complete_moderation_job"
RETRY_JOB_RPC = "retry_moderation_job"
FAIL_JOB_RPC = "fail_moderation_job"
RECOVER_STUCK_JOBS_RPC = "recover_stuck_moderation_jobs"
QUEUE_METRICS_RPC = "get_queue_metrics"
DECISION_COUNTS_TODAY_RPC = "get_moderation_decision_counts_today"
RETRY_BACKOFF_MINUTES = {
    1: 1,
    2: 5,
    3: 15,
}


class QueueServiceError(RuntimeError):
    """Raised when a moderation queue database operation fails."""


def _first_row(response: Any) -> dict[str, Any] | None:
    """Return the first Supabase response row, if present."""

    data = getattr(response, "data", None)
    if not data:
        return None
    return data[0]


def create_job(post_id: str, image_url: str) -> dict[str, Any]:
    """Create a pending moderation job and return the inserted row."""

    payload = {
        "post_id": post_id,
        "image_url": image_url,
        "status": "PENDING",
        "retry_count": 0,
        "max_retries": 3,
        "next_attempt_at": _now_iso(),
    }

    try:
        client = get_supabase_client()
        response = execute_with_retries(
            lambda: client.table(JOB_TABLE).insert(payload).execute(),
            action="create_moderation_job",
        )
        job = _first_row(response)
        if job is None:
            raise QueueServiceError("Supabase returned no row for created job.")
        logger.info("Created moderation job %s for post %s", job.get("id"), post_id)
        return job
    except QueueServiceError:
        raise
    except Exception as exc:
        logger.exception("Failed to create moderation job for post %s", post_id)
        raise QueueServiceError("Failed to create moderation job.") from exc


def get_next_pending_job() -> dict[str, Any] | None:
    """Return the oldest pending moderation job, or None when the queue is empty."""

    try:
        client = get_supabase_client()
        response = execute_with_retries(
            lambda: (
                client.table(JOB_TABLE)
                .select("*")
                .eq("status", "PENDING")
                .order("created_at", desc=False)
                .limit(1)
                .execute()
            ),
            action="get_next_pending_job",
        )
        job = _first_row(response)
        if job is None:
            logger.debug("No pending moderation jobs found.")
        else:
            logger.info("Fetched pending moderation job %s", job.get("id"))
        return job
    except Exception as exc:
        logger.exception("Failed to fetch next pending moderation job")
        raise QueueServiceError("Failed to fetch next pending moderation job.") from exc


def claim_next_pending_job(worker_id: str) -> dict[str, Any] | None:
    """Atomically claim the oldest pending moderation job for a worker.

    The backing RPC performs SELECT ... FOR UPDATE SKIP LOCKED and updates the
    chosen job to PROCESSING in one database transaction. This prevents multiple
    workers from receiving the same pending job.
    """

    try:
        client = get_supabase_client()
        response = execute_with_retries(
            lambda: client.rpc(CLAIM_JOB_RPC, {"p_worker_id": worker_id}).execute(),
            action=CLAIM_JOB_RPC,
        )
        job = _first_row(response)
        if job is None:
            logger.debug("No pending moderation jobs available to claim.")
        else:
            logger.info(
                "Worker %s claimed moderation job %s",
                worker_id,
                job.get("id"),
            )
        return job
    except Exception as exc:
        logger.exception("Failed to claim next pending moderation job")
        raise QueueServiceError("Failed to claim next pending moderation job.") from exc


def mark_processing(job_id: str, worker_id: str) -> dict[str, Any]:
    """Mark a moderation job as processing and assign it to a worker."""

    values = {
        "status": "PROCESSING",
        "worker_id": worker_id,
        "started_at": _now_iso(),
    }
    return _update_job(job_id, values, "mark moderation job as processing")


def mark_completed(job_id: str) -> dict[str, Any]:
    """Mark a moderation job as completed."""

    values = {
        "status": "COMPLETED",
        "completed_at": _now_iso(),
    }
    return _update_job(job_id, values, "mark moderation job as completed")


def mark_under_review(job_id: str) -> dict[str, Any]:
    """Mark a moderation job as requiring human review."""

    values = {
        "status": "UNDER_REVIEW",
        "completed_at": _now_iso(),
    }
    return _update_job(job_id, values, "mark moderation job under review")


def mark_failed(job_id: str, error_message: str) -> dict[str, Any]:
    """Mark a moderation job as failed with a stored error message."""

    values = {
        "status": "FAILED",
        "error_message": error_message,
        "completed_at": _now_iso(),
    }
    return _update_job(job_id, values, "mark moderation job as failed")


def complete_job_with_result(
    *,
    job_id: str,
    worker_id: str,
    post_id: str,
    scores: dict[str, float],
    category: str,
    confidence: float,
    decision: str,
    reason: str,
    ocr_text: str,
) -> dict[str, Any]:
    """Atomically persist moderation output and close the claimed queue job."""

    payload = {
        "p_job_id": job_id,
        "p_worker_id": worker_id,
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

    try:
        client = get_supabase_client()
        response = execute_with_retries(
            lambda: client.rpc(COMPLETE_JOB_RPC, payload).execute(),
            action=COMPLETE_JOB_RPC,
        )
        job = _first_row(response)
        if job is None:
            raise QueueServiceError(
                f"Supabase returned no row while completing job {job_id}."
            )
        return job
    except QueueServiceError:
        raise
    except Exception as exc:
        logger.exception("Failed to atomically complete moderation job %s", job_id)
        raise QueueServiceError("Failed to complete moderation job.") from exc


def retry_job_after_failure(
    job_id: str,
    worker_id: str,
    error_message: str,
    next_attempt_at: str,
) -> int:
    """Atomically increment retry count and return the job to pending."""

    try:
        client = get_supabase_client()
        response = execute_with_retries(
            lambda: client.rpc(
                RETRY_JOB_RPC,
                {
                    "p_job_id": job_id,
                    "p_worker_id": worker_id,
                    "p_error_message": error_message,
                    "p_next_attempt_at": next_attempt_at,
                },
            ).execute(),
            action=RETRY_JOB_RPC,
        )
        row = _first_row(response)
        if row is None:
            raise QueueServiceError(
                f"Supabase returned no row while retrying job {job_id}."
            )
        return int(row.get("retry_count") or 0)
    except QueueServiceError:
        raise
    except Exception as exc:
        logger.exception("Failed to retry moderation job %s", job_id)
        raise QueueServiceError("Failed to retry moderation job.") from exc


def fail_job(job_id: str, worker_id: str, error_message: str) -> dict[str, Any]:
    """Atomically mark a claimed moderation job as permanently failed."""

    try:
        client = get_supabase_client()
        response = execute_with_retries(
            lambda: client.rpc(
                FAIL_JOB_RPC,
                {
                    "p_job_id": job_id,
                    "p_worker_id": worker_id,
                    "p_error_message": error_message,
                },
            ).execute(),
            action=FAIL_JOB_RPC,
        )
        job = _first_row(response)
        if job is None:
            raise QueueServiceError(
                f"Supabase returned no row while failing job {job_id}."
            )
        return job
    except QueueServiceError:
        raise
    except Exception as exc:
        logger.exception("Failed to fail moderation job %s", job_id)
        raise QueueServiceError("Failed to fail moderation job.") from exc


def recover_stuck_jobs(cutoff_iso: str) -> int:
    """Atomically recover stale processing jobs and return the recovery count."""

    try:
        client = get_supabase_client()
        response = execute_with_retries(
            lambda: client.rpc(
                RECOVER_STUCK_JOBS_RPC,
                {"p_cutoff": cutoff_iso},
            ).execute(),
            action=RECOVER_STUCK_JOBS_RPC,
        )
        row = _first_row(response) or {}
        return int(row.get("recovered_count") or 0)
    except Exception as exc:
        logger.exception("Failed to recover stuck moderation jobs")
        raise QueueServiceError("Failed to recover stuck moderation jobs.") from exc


def increment_retry(job_id: str) -> int:
    """Increment a moderation job retry counter and return the updated count."""

    try:
        client = get_supabase_client()
        response = execute_with_retries(
            lambda: (
                client.table(JOB_TABLE)
                .select("retry_count")
                .eq("id", job_id)
                .limit(1)
                .execute()
            ),
            action="read_moderation_job_retry_count",
        )
        job = _first_row(response)
        if job is None:
            raise QueueServiceError(f"Moderation job {job_id} was not found.")

        current_retry_count = int(job.get("retry_count") or 0)
        updated_retry_count = current_retry_count + 1

        update_response = execute_with_retries(
            lambda: (
                client.table(JOB_TABLE)
                .update({"retry_count": updated_retry_count})
                .eq("id", job_id)
                .execute()
            ),
            action="increment_moderation_job_retry_count",
        )
        updated_job = _first_row(update_response)
        if updated_job is None:
            raise QueueServiceError(
                f"Supabase returned no row after retry update for job {job_id}."
            )

        logger.info(
            "Incremented retry count for moderation job %s to %s",
            job_id,
            updated_retry_count,
        )
        return int(updated_job.get("retry_count", updated_retry_count))
    except QueueServiceError:
        raise
    except Exception as exc:
        logger.exception("Failed to increment retry count for moderation job %s", job_id)
        raise QueueServiceError("Failed to increment moderation job retry count.") from exc


def retry_backoff_until(retry_count: int) -> str:
    """Return the next eligible attempt timestamp for a retry count."""

    minutes = RETRY_BACKOFF_MINUTES.get(retry_count, 15)
    return (datetime.now(UTC) + timedelta(minutes=minutes)).isoformat()


def get_queue_depth_by_status() -> dict[str, int]:
    """Return moderation job counts grouped by status."""

    try:
        client = get_supabase_client()
        response = execute_with_retries(
            lambda: client.rpc(QUEUE_METRICS_RPC, {}).execute(),
            action=QUEUE_METRICS_RPC,
        )
        counts = {
            "PENDING": 0,
            "PROCESSING": 0,
            "UNDER_REVIEW": 0,
            "FAILED": 0,
            "COMPLETED": 0,
        }
        for row in getattr(response, "data", None) or []:
            status = str(row.get("status") or "").upper()
            if status:
                counts[status] = int(row.get("count") or 0)
        return counts
    except Exception as exc:
        logger.exception("Failed to fetch moderation queue depth")
        raise QueueServiceError("Failed to fetch moderation queue depth.") from exc


def get_decision_counts_today() -> dict[str, int]:
    """Return today's approved/rejected moderation decision counts from DB RPC."""

    try:
        client = get_supabase_client()
        response = execute_with_retries(
            lambda: client.rpc(DECISION_COUNTS_TODAY_RPC, {}).execute(),
            action=DECISION_COUNTS_TODAY_RPC,
        )
        row = _first_row(response) or {}
        return {
            "approved_today": int(row.get("approved_today") or 0),
            "rejected_today": int(row.get("rejected_today") or 0),
        }
    except Exception as exc:
        logger.exception("Failed to fetch moderation decision counts")
        raise QueueServiceError("Failed to fetch moderation decision counts.") from exc


def get_oldest_pending_job_created_at() -> str | None:
    """Return the oldest pending job timestamp, if any."""

    try:
        client = get_supabase_client()
        response = execute_with_retries(
            lambda: (
                client.table(JOB_TABLE)
                .select("created_at")
                .eq("status", "PENDING")
                .order("created_at", desc=False)
                .limit(1)
                .execute()
            ),
            action="get_oldest_pending_job_created_at",
        )
        job = _first_row(response)
        return str(job["created_at"]) if job and job.get("created_at") else None
    except Exception as exc:
        logger.exception("Failed to fetch oldest pending moderation job")
        raise QueueServiceError("Failed to fetch oldest pending moderation job.") from exc


def _update_job(job_id: str, values: dict[str, Any], action: str) -> dict[str, Any]:
    """Update a moderation job and return the updated row."""

    try:
        client = get_supabase_client()
        response = execute_with_retries(
            lambda: client.table(JOB_TABLE).update(values).eq("id", job_id).execute(),
            action=action,
        )
        job = _first_row(response)
        if job is None:
            raise QueueServiceError(
                f"Supabase returned no row while trying to {action}."
            )
        logger.info("Updated moderation job %s: %s", job_id, action)
        return job
    except QueueServiceError:
        raise
    except Exception as exc:
        logger.exception("Failed to %s for job %s", action, job_id)
        raise QueueServiceError(f"Failed to {action}.") from exc


def _now_iso() -> str:
    """Return the current UTC timestamp in a Supabase-compatible format."""

    return datetime.now(UTC).isoformat()
