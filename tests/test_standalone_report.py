from types import SimpleNamespace

from backend.reports import build_report, decide, normalize_categories


def test_normalize_categories_exposes_public_taxonomy():
    categories = normalize_categories(
        {
            "adult_score": 0.12,
            "weapon_score": 0.741,
            "blood_score": 18.2,
            "ml_toxicity_score": 0.33,
        }
    )

    assert categories["adult_content"] == 12.0
    assert categories["weapons"] == 74.1
    assert categories["blood"] == 18.2
    assert categories["toxic_text"] == 33.0
    assert "medical_content" in categories


def test_decision_engine_routes_high_risk_to_review():
    decision = decide(74.1, {"weapons": 74.1, "adult_content": 0, "child_safety_risk": 0})

    assert decision.risk_level == "HIGH RISK"
    assert decision.decision == "Review Required"
    assert decision.recommendation == "Human Review"


def test_build_report_from_pipeline_result():
    result = SimpleNamespace(
        scores={"ensemble_risk_score": 0.832, "weapon_score": 0.741},
        category_scores={"blood_score": 0.182},
        detected_objects=[{"label": "knife"}, {"class": "person"}, "knife"],
        ocr_text="sample text",
        generated_caption="a person holding an object",
        image_hash="abc123",
        model_versions={"vision": "test"},
        pipeline_error=False,
        error_reason=None,
    )

    report = build_report(result)

    assert report["overall_score"] == 83.2
    assert report["risk_level"] == "HIGH RISK"
    assert report["decision"] == "Review Required"
    assert report["objects"] == ["knife", "person"]
    assert report["ocr_text"] == "sample text"
