from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_health_alias_succeeds() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_status_endpoint_contract() -> None:
    expected = {
        "nsfw",
        "siglip",
        "yolo",
        "ocr_surya",
        "blip",
        "llama",
        "text_classifier",
    }
    allowed = {"loaded", "failed", "disabled", "not_loaded", "lazy"}

    with TestClient(app) as client:
        response = client.get("/api/models/status")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == expected
    assert set(payload.values()).issubset(allowed)


def test_runtime_status_endpoint_contract() -> None:
    with TestClient(app) as client:
        response = client.get("/api/runtime/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] in {"cuda", "npu", "cpu"}
    assert payload["torch_device"]
    assert "execution_providers" in payload
