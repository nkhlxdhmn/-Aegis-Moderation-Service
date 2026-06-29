"""Worker heartbeat persistence and status helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from supabase_client import execute_with_retries, get_supabase_client

HEARTBEAT_TABLE = "worker_heartbeats"
ONLINE_THRESHOLD_SECONDS = 90


def record_worker_heartbeat(
    worker_id: str,
    jobs_processed: int,
    gpu_id: str | None = None,
    client: Any | None = None,
) -> None:
    db = client or get_supabase_client()
    payload = {
        "worker_id": worker_id,
        "last_seen": datetime.now(UTC).isoformat(),
        "jobs_processed": jobs_processed,
        "gpu_id": gpu_id,
    }
    execute_with_retries(
        lambda: db.table(HEARTBEAT_TABLE).upsert(payload, on_conflict="worker_id").execute(),
        action="record_worker_heartbeat",
    )


def get_worker_statuses(client: Any | None = None) -> list[dict[str, Any]]:
    db = client or get_supabase_client()
    response = db.table(HEARTBEAT_TABLE).select("*").execute()
    rows = getattr(response, "data", None) or []
    threshold = datetime.now(UTC) - timedelta(seconds=ONLINE_THRESHOLD_SECONDS)
    statuses = []
    for row in rows:
        last_seen = _parse_timestamp(row.get("last_seen"))
        statuses.append(
            {
                **row,
                "online": last_seen is not None and last_seen >= threshold,
            }
        )
    return statuses


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
