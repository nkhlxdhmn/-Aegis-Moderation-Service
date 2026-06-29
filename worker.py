"""Background moderation worker for queued moderation jobs.

The worker orchestrates existing service components only. It does not define
new moderation models, thresholds, or decision rules.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from dotenv import load_dotenv

from main import (
    PIPELINE_ERROR_REASON,
    _download_image,
    _validate_image_file,
    _with_reason_code,
)
from pipeline.category_engine import get_top_category
from pipeline.decision_engine import decide_with_reason_code
from pipeline.safety_flags import analyze_image
import heartbeat_service
import model_warmup
import queue_service

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_WORKER_ID = "worker-local"
DEFAULT_POLL_INTERVAL_SECONDS = 5
DEFAULT_PROCESSING_TIMEOUT_MINUTES = 10
HEARTBEAT_INTERVAL_SECONDS = 30
_jobs_processed = 0
_last_heartbeat_at = 0.0


def _worker_id() -> str:
    """Return the configured worker identifier."""

    return os.getenv("WORKER_ID") or DEFAULT_WORKER_ID


def _poll_interval_seconds() -> float:
    """Return the configured queue polling interval in seconds."""

    raw_value = os.getenv("POLL_INTERVAL_SECONDS")
    if not raw_value:
        return float(DEFAULT_POLL_INTERVAL_SECONDS)
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning(
            "Invalid POLL_INTERVAL_SECONDS=%r; using default %s",
            raw_value,
            DEFAULT_POLL_INTERVAL_SECONDS,
        )
        return float(DEFAULT_POLL_INTERVAL_SECONDS)
    return max(0.0, value)


def _processing_timeout_minutes() -> float:
    """Return the processing recovery timeout in minutes."""

    raw_value = os.getenv("PROCESSING_TIMEOUT_MINUTES")
    if not raw_value:
        return float(DEFAULT_PROCESSING_TIMEOUT_MINUTES)
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning(
            "Invalid PROCESSING_TIMEOUT_MINUTES=%r; using default %s",
            raw_value,
            DEFAULT_PROCESSING_TIMEOUT_MINUTES,
        )
        return float(DEFAULT_PROCESSING_TIMEOUT_MINUTES)
    return max(0.0, value)


def process_single_job() -> bool:
    """Process one pending moderation job.

    Returns True when a job was found and handled. Returns False when the queue
    is empty so callers can sleep before polling again.
    """

    worker_id = _worker_id()
    _record_heartbeat_if_due(worker_id)
    job = queue_service.claim_next_pending_job(worker_id)
    if job is None:
        logger.debug("Moderation queue is empty")
        return False

    job_id = str(job["id"])
    post_id = str(job["post_id"])
    image_url = str(job["image_url"])
    current_retry_count = int(job.get("retry_count") or 0)
    max_retries = int(job.get("max_retries") or 3)
    request_id = str(uuid4())

    logger.info(
        "Worker picked moderation job",
        extra={
            "request_id": request_id,
            "worker_id": worker_id,
            "job_id": job_id,
            "post_id": post_id,
        },
    )

    temp_image_path: str | None = None
    try:
        temp_image_path = _download_image(image_url)
        _validate_image_file(temp_image_path)
        pipeline_result = analyze_image(temp_image_path, job.get("caption"))

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

        queue_service.complete_job_with_result(
            job_id=job_id,
            worker_id=worker_id,
            post_id=post_id,
            scores=scores,
            category=category,
            confidence=category_confidence,
            decision=decision,
            reason=reason,
            ocr_text=ocr_text,
        )

        if decision == "UNDER_REVIEW":
            logger.info(
                "Moderation job completed under review",
                extra={
                    "request_id": request_id,
                    "worker_id": worker_id,
                    "job_id": job_id,
                    "post_id": post_id,
                    "decision": decision,
                },
            )
        else:
            logger.info(
                "Moderation job completed",
                extra={
                    "request_id": request_id,
                    "worker_id": worker_id,
                    "job_id": job_id,
                    "post_id": post_id,
                    "decision": decision,
                },
            )

        _record_job_processed(worker_id)
        return True
    except Exception as exc:
        _handle_job_failure(job_id, worker_id, current_retry_count, max_retries, exc)
        return True
    finally:
        if temp_image_path is not None:
            Path(temp_image_path).unlink(missing_ok=True)


def recover_stuck_jobs() -> int:
    """Return stale processing jobs to the pending queue after a worker crash."""

    cutoff = datetime.now(UTC) - timedelta(minutes=_processing_timeout_minutes())
    try:
        return queue_service.recover_stuck_jobs(cutoff.isoformat())
    except Exception as exc:
        logger.exception("Failed to recover stuck moderation jobs")
        raise queue_service.QueueServiceError(
            "Failed to recover stuck moderation jobs."
        ) from exc


def run_worker() -> None:
    """Run the moderation worker loop forever."""

    worker_id = _worker_id()
    poll_interval = _poll_interval_seconds()
    logger.info(
        "Moderation worker starting: worker_id=%s poll_interval_seconds=%s",
        worker_id,
        poll_interval,
    )

    model_warmup.warmup_models_if_enabled()
    try:
        recovered_count = recover_stuck_jobs()
        logger.info("Recovered %s stuck jobs", recovered_count)
    except Exception:
        logger.warning(
            "Stuck-job recovery failed at startup (schema migration pending?); "
            "worker will continue processing new jobs."
        )
    _record_heartbeat(worker_id)

    while True:
        try:
            handled_job = process_single_job()
        except Exception:
            # process_single_job already logs per-job failures.  An exception
            # escaping here means failure-handling itself failed (e.g. Supabase
            # was unreachable while trying to retry or fail the job).  Sleep and
            # continue so one bad job cannot kill the worker permanently.
            logger.exception(
                "Unhandled error in moderation worker main loop; sleeping before retry."
            )
            time.sleep(poll_interval)
            continue
        if not handled_job:
            time.sleep(poll_interval)


def _handle_job_failure(
    job_id: str,
    worker_id: str,
    current_retry_count: int,
    max_retries: int,
    exc: Exception,
) -> None:
    """Retry or permanently fail a job after an exception."""

    logger.exception("Moderation job %s failed during processing", job_id)
    error_message = str(exc)
    retry_count = current_retry_count + 1

    if retry_count < max_retries:
        next_attempt_at = queue_service.retry_backoff_until(retry_count)
        retry_count = queue_service.retry_job_after_failure(
            job_id,
            worker_id,
            error_message,
            next_attempt_at,
        )
        logger.info(
            "Moderation job %s scheduled for retry %s/%s at %s",
            job_id,
            retry_count,
            max_retries,
            next_attempt_at,
        )
        return

    queue_service.fail_job(job_id, worker_id, error_message)
    logger.error(
        "Moderation job %s failed permanently after %s/%s retries",
        job_id,
        retry_count,
        max_retries,
    )


def _record_heartbeat_if_due(worker_id: str) -> None:
    if time.time() - _last_heartbeat_at >= HEARTBEAT_INTERVAL_SECONDS:
        _record_heartbeat(worker_id)


def _record_heartbeat(worker_id: str) -> None:
    global _last_heartbeat_at
    try:
        heartbeat_service.record_worker_heartbeat(
            worker_id,
            jobs_processed=_jobs_processed,
            gpu_id=os.getenv("GPU_ID"),
        )
        _last_heartbeat_at = time.time()
        # Update the docker health-check sentinel file.
        try:
            Path("/tmp/worker_alive").touch()
        except OSError:
            pass
    except Exception:
        logger.exception("Failed to record worker heartbeat")


def _record_job_processed(worker_id: str) -> None:
    global _jobs_processed
    _jobs_processed += 1
    _record_heartbeat(worker_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_worker()
