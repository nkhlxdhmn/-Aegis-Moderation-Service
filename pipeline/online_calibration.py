"""Temporal threshold calibration from admin review feedback.

Reads admin_audit_log + moderation_results from Supabase to compute empirical
false-positive and false-negative rates, then writes conservative threshold
adjustments to a JSON config file.

Design:
  - Admin decision ground-truth labels:
      ADMIN_APPROVED after pipeline REJECTED → false positive
      ADMIN_REJECTED after pipeline APPROVED → false negative
      ADMIN_REJECTED after pipeline UNDER_REVIEW → late catch (counted as near-miss)
      ADMIN_APPROVED after pipeline UNDER_REVIEW → false positive from review queue
  - Threshold adjustment: capped at ±0.02 per calibration run.
  - Never auto-applies adjustments > ±0.02 to prevent runaway calibration.
  - Adjustments are stored in CALIBRATION_FILE (JSON) and loaded by the service
    on startup to override code defaults without code changes.

Usage:
    python -m pipeline.online_calibration --run
    python -m pipeline.online_calibration --dry-run
    python -m pipeline.online_calibration --status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path(os.getenv("CALIBRATION_FILE", "/app/data/calibration.json"))
MAX_ADJUSTMENT_PER_RUN = 0.02  # hard cap per calibration run
TARGET_FP_RATE = 0.03          # goal: false positives < 3 %
TARGET_FN_RATE = 0.01          # goal: false negatives < 1 %

# Only consider admin actions from the last N days (rolling window)
CALIBRATION_WINDOW_DAYS = 30


def _get_client():
    from supabase_client import get_supabase_client
    return get_supabase_client()


def _load_calibration() -> dict[str, Any]:
    """Load existing calibration config, or return empty defaults."""
    if CALIBRATION_FILE.exists():
        try:
            return json.loads(CALIBRATION_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read calibration file; using defaults")
    return {
        "last_run": None,
        "fp_rate": None,
        "fn_rate": None,
        "adjustments": {},
        "history": [],
    }


def _save_calibration(data: dict[str, Any]) -> None:
    """Write calibration config to disk, creating parent dirs if needed."""
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Calibration config saved to %s", CALIBRATION_FILE)


def fetch_admin_actions(days: int = CALIBRATION_WINDOW_DAYS) -> list[dict]:
    """Fetch admin review actions joined with original pipeline decisions.

    Returns rows with keys:
      post_id, pipeline_decision, admin_decision, created_at
    """
    client = _get_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        resp = (
            client.table("admin_audit_log")
            .select("post_id, action, created_at, moderation_results(decision)")
            .gte("created_at", cutoff)
            .in_("action", ["APPROVED", "REJECTED"])
            .execute()
        )
        rows = getattr(resp, "data", []) or []
        result = []
        for row in rows:
            mr = (row.get("moderation_results") or {})
            if isinstance(mr, list):
                mr = mr[0] if mr else {}
            result.append({
                "post_id": row.get("post_id"),
                "pipeline_decision": mr.get("decision", "UNKNOWN"),
                "admin_decision": row.get("action"),
                "created_at": row.get("created_at"),
            })
        return result
    except Exception:
        logger.exception("Failed to fetch admin actions from Supabase")
        return []


def compute_error_rates(
    actions: list[dict],
) -> tuple[float, float, dict[str, int]]:
    """Return (fp_rate, fn_rate, counts).

    Definitions:
      false positive  — pipeline REJECTED but admin APPROVED
      false negative  — pipeline APPROVED but admin REJECTED
      true positive   — pipeline REJECTED/UNDER_REVIEW, admin REJECTED
    """
    counts: dict[str, int] = {
        "fp": 0, "fn": 0, "tp": 0, "tn": 0,
        "total_admin_approved": 0, "total_admin_rejected": 0,
    }

    for row in actions:
        pipe = str(row.get("pipeline_decision", "")).upper()
        admin = str(row.get("admin_decision", "")).upper()

        if admin == "APPROVED":
            counts["total_admin_approved"] += 1
            if pipe == "REJECTED":
                counts["fp"] += 1       # over-blocked
            elif pipe in ("APPROVED", "UNDER_REVIEW"):
                counts["tn"] += 1
        elif admin == "REJECTED":
            counts["total_admin_rejected"] += 1
            if pipe == "APPROVED":
                counts["fn"] += 1       # missed harmful content
            elif pipe in ("REJECTED", "UNDER_REVIEW"):
                counts["tp"] += 1

    fp_rate = (
        counts["fp"] / counts["total_admin_approved"]
        if counts["total_admin_approved"] > 0 else 0.0
    )
    fn_rate = (
        counts["fn"] / counts["total_admin_rejected"]
        if counts["total_admin_rejected"] > 0 else 0.0
    )
    return fp_rate, fn_rate, counts


def suggest_adjustments(fp_rate: float, fn_rate: float) -> dict[str, float]:
    """Suggest threshold delta values to nudge fp/fn rates toward targets.

    High FP rate (> TARGET_FP_RATE) → thresholds too low → increase them.
    High FN rate (> TARGET_FN_RATE) → thresholds too high → decrease them.
    Never suggests > MAX_ADJUSTMENT_PER_RUN in magnitude.

    Returns a dict of {threshold_name: delta}.  Positive delta = raise threshold.
    """
    adjustments: dict[str, float] = {}

    fp_excess = fp_rate - TARGET_FP_RATE
    fn_excess = fn_rate - TARGET_FN_RATE

    if fp_excess > 0 and fn_excess > 0:
        # Both rates elevated: precision-recall trade-off — prefer FN reduction.
        logger.warning(
            "Both FP (%.1f%%) and FN (%.1f%%) rates elevated — prioritising FN reduction",
            fp_rate * 100, fn_rate * 100,
        )
        # Scale adjustment proportionally, capped
        delta = min(MAX_ADJUSTMENT_PER_RUN, fn_excess * 0.5)
        adjustments["ADULT_REJECT_THRESHOLD"] = -delta
        adjustments["VIOLENCE_SELF_HARM_THRESHOLD"] = -delta
        adjustments["WEAPON_THRESHOLD"] = -delta
    elif fp_excess > 0:
        # Too many false positives → raise thresholds (less aggressive blocking)
        delta = min(MAX_ADJUSTMENT_PER_RUN, fp_excess * 0.5)
        adjustments["ADULT_REJECT_THRESHOLD"] = +delta
        adjustments["VIOLENCE_SELF_HARM_THRESHOLD"] = +delta
        adjustments["TERRORISM_THRESHOLD"] = +delta
        adjustments["FRAUD_THRESHOLD"] = +delta
    elif fn_excess > 0:
        # Too many false negatives → lower thresholds (catch more content)
        delta = min(MAX_ADJUSTMENT_PER_RUN, fn_excess * 0.5)
        adjustments["ADULT_REJECT_THRESHOLD"] = -delta
        adjustments["VIOLENCE_SELF_HARM_THRESHOLD"] = -delta
        adjustments["CHILD_SAFETY_REVIEW_THRESHOLD"] = -delta
        adjustments["FRAUD_THRESHOLD"] = -delta

    # Round to 4 dp for readability
    return {k: round(v, 4) for k, v in adjustments.items()}


def apply_calibration(
    adjustments: dict[str, float],
    current: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply adjustments to the calibration config and return the updated config.

    Skips silently if no adjustments are suggested.
    When dry_run=True, logs what would change without writing to disk.
    """
    if not adjustments:
        logger.info("No threshold adjustments suggested — pipeline is within targets")
        return current

    existing_adj: dict[str, float] = current.get("adjustments", {})

    for key, delta in adjustments.items():
        prev = existing_adj.get(key, 0.0)
        new_val = round(prev + delta, 4)
        # Hard limit: cumulative adjustment from code baseline capped at ±0.10
        clamped = max(-0.10, min(0.10, new_val))
        if dry_run:
            logger.info("[DRY-RUN] %s: %.4f → %.4f (delta=%.4f)", key, prev, clamped, delta)
        else:
            existing_adj[key] = clamped
            logger.info("Adjusted %s: %.4f → %.4f", key, prev, clamped)

    if not dry_run:
        current["adjustments"] = existing_adj

    return current


def run_calibration(*, dry_run: bool = False) -> dict[str, Any]:
    """Full calibration pipeline: fetch → compute → suggest → apply → save.

    Returns the updated calibration config.
    """
    logger.info("Starting online calibration (dry_run=%s)", dry_run)

    actions = fetch_admin_actions()
    if not actions:
        logger.warning("No admin actions found — calibration skipped")
        return _load_calibration()

    fp_rate, fn_rate, counts = compute_error_rates(actions)
    logger.info(
        "Calibration stats: FP=%.1f%% FN=%.1f%% TP=%d FN_count=%d "
        "(window=%d days, n=%d admin actions)",
        fp_rate * 100, fn_rate * 100,
        counts["tp"], counts["fn"],
        CALIBRATION_WINDOW_DAYS, len(actions),
    )

    adjustments = suggest_adjustments(fp_rate, fn_rate)
    config = _load_calibration()

    config["last_run"] = datetime.now(timezone.utc).isoformat()
    config["fp_rate"] = round(fp_rate, 4)
    config["fn_rate"] = round(fn_rate, 4)
    config["counts"] = counts

    history_entry = {
        "ts": config["last_run"],
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "adjustments": adjustments,
        "dry_run": dry_run,
    }
    config.setdefault("history", []).append(history_entry)
    # Keep last 52 entries (~1 year of weekly runs)
    config["history"] = config["history"][-52:]

    config = apply_calibration(adjustments, config, dry_run=dry_run)

    if not dry_run:
        _save_calibration(config)

    return config


def load_threshold_overrides() -> dict[str, float]:
    """Load threshold override deltas from calibration file.

    Called by decision_engine at startup.  Returns {} if no calibration exists.
    The caller should ADD these deltas to the code-defined baseline thresholds.
    """
    config = _load_calibration()
    return config.get("adjustments", {})


def get_status() -> dict[str, Any]:
    """Return human-readable calibration status."""
    config = _load_calibration()
    return {
        "calibration_file": str(CALIBRATION_FILE),
        "last_run": config.get("last_run"),
        "fp_rate": config.get("fp_rate"),
        "fn_rate": config.get("fn_rate"),
        "active_adjustments": config.get("adjustments", {}),
        "history_entries": len(config.get("history", [])),
        "targets": {
            "fp_rate_target": TARGET_FP_RATE,
            "fn_rate_target": TARGET_FN_RATE,
            "max_adjustment_per_run": MAX_ADJUSTMENT_PER_RUN,
        },
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    parser = argparse.ArgumentParser(description="Online threshold calibration")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="Run calibration and write adjustments")
    group.add_argument("--dry-run", action="store_true", help="Show adjustments without writing")
    group.add_argument("--status", action="store_true", help="Show current calibration status")
    args = parser.parse_args()

    if args.status:
        import json as _json
        print(_json.dumps(get_status(), indent=2, default=str))
    else:
        result = run_calibration(dry_run=args.dry_run)
        print(f"Calibration complete. FP={result.get('fp_rate', 'N/A')} FN={result.get('fn_rate', 'N/A')}")
        if result.get("adjustments"):
            print("Active adjustments:")
            for k, v in result["adjustments"].items():
                print(f"  {k}: {v:+.4f}")
