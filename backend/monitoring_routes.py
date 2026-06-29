"""Monitoring and observability API routes for Aegis Moderation.

Registers an APIRouter under /api/v1/monitor that the main FastAPI app
includes.  All endpoints read from the in-process AegisMonitor singleton
and return JSON (or streaming CSV for exports).
"""

from __future__ import annotations

import csv
import io
import json
import time
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from backend.monitor import monitor

router = APIRouter(prefix="/api/v1/monitor", tags=["monitoring"])


@router.get("/system")
def system_metrics() -> JSONResponse:
    """CPU, memory, GPU, disk, network, uptime."""
    return JSONResponse(monitor.get_system_stats())


@router.get("/requests")
def request_analytics() -> JSONResponse:
    """Total requests, success/fail counts, RPS, response times, history."""
    return JSONResponse(monitor.get_request_analytics())


@router.get("/moderation")
def moderation_analytics() -> JSONResponse:
    """Requests by content type, moderation decisions, category flag counts."""
    return JSONResponse(monitor.get_moderation_analytics())


@router.get("/models")
def model_stats() -> JSONResponse:
    """Model load status, load times, per-model inference breakdown."""
    return JSONResponse(monitor.get_model_stats())


@router.get("/performance")
def performance_stats() -> JSONResponse:
    """End-to-end latency, OCR/vision/NLP averages, throughput."""
    return JSONResponse(monitor.get_performance_stats())


@router.get("/health")
def health_dashboard() -> JSONResponse:
    """Per-component health status (healthy / warning / offline)."""
    return JSONResponse(monitor.get_health())


@router.get("/errors")
def error_dashboard() -> JSONResponse:
    """Failed requests grouped by error type with recent detail."""
    return JSONResponse(monitor.get_errors())


@router.get("/security")
def security_dashboard() -> JSONResponse:
    """Security events: blocked file types, oversized uploads, SSRF attempts."""
    return JSONResponse(monitor.get_security_events())


@router.get("/logs")
def log_viewer(
    level: str | None = Query(
        default=None, description="Filter by log level (INFO, WARNING, ERROR, DEBUG)"
    ),
    search: str | None = Query(default=None, description="Search term (case-insensitive)"),
    limit: int = Query(default=200, le=1000, description="Maximum lines to return"),
) -> JSONResponse:
    """Recent application logs with optional level/search filtering."""
    return JSONResponse({"logs": monitor.get_logs(level=level, search=search, limit=limit)})


@router.get("/all")
def all_metrics() -> JSONResponse:
    """All monitoring data in one call — used by the dashboard on each refresh."""
    return JSONResponse(
        {
            "ts": time.time(),
            "system": monitor.get_system_stats(),
            "requests": monitor.get_request_analytics(),
            "moderation": monitor.get_moderation_analytics(),
            "models": monitor.get_model_stats(),
            "performance": monitor.get_performance_stats(),
            "health": monitor.get_health(),
            "errors": monitor.get_errors(),
            "security": monitor.get_security_events(),
        }
    )


@router.get("/export")
def export_data(
    format: str = Query(default="json", description="Export format: json or csv"),
) -> Any:
    """Export all monitoring data as JSON or CSV download."""
    data = monitor.export_all()

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["metric", "value"])

        sys_s = data["system"]
        writer.writerow(["uptime_seconds", sys_s.get("uptime_seconds", "")])
        writer.writerow(["cpu_percent", sys_s.get("cpu_percent", "")])
        mem = sys_s.get("memory") or {}
        writer.writerow(["memory_used_gb", mem.get("used_gb", "")])
        writer.writerow(["memory_percent", mem.get("percent", "")])
        disk = sys_s.get("disk") or {}
        writer.writerow(["disk_used_gb", disk.get("used_gb", "")])
        writer.writerow(["disk_percent", disk.get("percent", "")])
        writer.writerow(["active_threads", sys_s.get("active_threads", "")])

        req = data["requests"]
        writer.writerow(["total_requests", req.get("total", "")])
        writer.writerow(["successful_requests", req.get("successful", "")])
        writer.writerow(["failed_requests", req.get("failed", "")])
        writer.writerow(["avg_response_time_s", req.get("avg_response_time_s", "")])
        writer.writerow(["requests_per_second", req.get("requests_per_second", "")])

        mod = data["moderation"]
        for ct, count in (mod.get("by_content_type") or {}).items():
            writer.writerow([f"content_type_{ct}", count])
        for dec, count in (mod.get("decisions") or {}).items():
            writer.writerow([f"decision_{dec}", count])
        for cat, count in (mod.get("category_flags") or {}).items():
            writer.writerow([f"category_{cat}", count])

        perf = data["performance"]
        writer.writerow(["avg_e2e_latency_s", perf.get("avg_e2e_latency_s", "")])
        writer.writerow(["avg_ocr_time_s", perf.get("avg_ocr_time_s", "")])
        writer.writerow(["avg_vision_time_s", perf.get("avg_vision_time_s", "")])
        writer.writerow(["avg_nlp_time_s", perf.get("avg_nlp_time_s", "")])
        writer.writerow(["throughput_per_minute", perf.get("throughput_per_minute", "")])

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=aegis_metrics.csv"},
        )

    # Default: JSON
    payload = json.dumps(data, default=str)
    return StreamingResponse(
        iter([payload]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=aegis_metrics.json"},
    )
