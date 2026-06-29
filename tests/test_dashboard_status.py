from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app


def test_dashboard_reads_live_model_status_endpoint() -> None:
    html = Path("frontend/dashboard.html").read_text(encoding="utf-8")

    assert "/api/models/status" in html
    assert "const REFRESH_MS = 5000" in html
    assert "lazy load" in html


def test_monitor_all_models_match_status_endpoint_keys() -> None:
    with TestClient(app) as client:
        model_status = client.get("/api/models/status").json()
        monitor_all = client.get("/api/v1/monitor/all").json()

    monitor_status = monitor_all["models"]["model_status"]
    assert set(monitor_status) == set(model_status)
