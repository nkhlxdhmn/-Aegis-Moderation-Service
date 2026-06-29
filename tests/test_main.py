from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root_serves_dashboard():
    response = client.get("/")

    assert response.status_code == 200
    assert "Aegis Moderation" in response.text


def test_health_is_versioned_and_standalone():
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["mode"] == "standalone"


def test_model_health_documents_lazy_loading():
    response = client.get("/api/v1/model-health")

    assert response.status_code == 200
    assert response.json()["mode"] == "lazy-load"


def test_analyze_requires_exactly_one_input():
    response = client.post("/api/v1/analyze", data={})

    assert response.status_code == 400


def test_moderate_url_returns_report(tmp_path):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake")
    pipeline_result = SimpleNamespace(
        scores={"ensemble_risk_score": 0.1},
        category_scores={},
        detected_objects=[],
        ocr_text="",
        generated_caption="",
        image_hash=None,
        model_versions={},
        pipeline_error=False,
        error_reason=None,
    )

    with (
        patch("main.download_image", return_value=image_path),
        patch("main.analyze_image", return_value=pipeline_result),
    ):
        response = client.post(
            "/api/v1/moderate", json={"image_url": "https://example.com/image.jpg"}
        )

    assert response.status_code == 200
    assert response.json()["decision"] == "Accept"
