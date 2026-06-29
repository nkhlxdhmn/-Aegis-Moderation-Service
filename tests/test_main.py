from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app

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
        patch("backend.main.download_image", return_value=image_path),
        patch("backend.main.analyze_image", return_value=pipeline_result),
    ):
        response = client.post(
            "/api/v1/moderate", json={"image_url": "https://example.com/image.jpg"}
        )

    assert response.status_code == 200
    assert response.json()["decision"] == "Accept"


def test_text_moderation_endpoint_returns_report():
    pipeline_result = SimpleNamespace(
        scores={"ensemble_risk_score": 0.2, "spam_score": 0.2},
        category_scores={},
        detected_objects=[],
        ocr_text="",
        generated_caption="",
        image_hash=None,
        model_versions={},
        pipeline_error=False,
        error_reason=None,
    )

    with patch("backend.main.moderate_text", return_value=pipeline_result):
        response = client.post("/api/v1/moderate/text", json={"text": "hello world"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["content_type"] == "text"
    assert payload["extracted_text_preview"] == "hello world"


def test_pdf_requires_exactly_one_input():
    response = client.post("/api/v1/moderate/pdf", data={})

    assert response.status_code == 400


def test_docx_requires_exactly_one_input():
    response = client.post("/api/v1/moderate/docx", data={})

    assert response.status_code == 400


def test_pdf_upload_uses_document_processor(tmp_path):
    document = SimpleNamespace(path=tmp_path / "upload.pdf")
    document.path.write_bytes(b"%PDF")
    report = {
        "overall_score": 0.0,
        "risk_level": "SAFE",
        "decision": "Accept",
        "recommendation": "Allow",
        "categories": {},
        "objects": [],
        "ocr_text": "",
    }

    with (
        patch("backend.main.write_document_upload", return_value=document),
        patch("backend.main.moderate_pdf", return_value=report),
    ):
        response = client.post(
            "/api/v1/moderate/pdf",
            files={"file": ("sample.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json()["content_type"] == "pdf"
