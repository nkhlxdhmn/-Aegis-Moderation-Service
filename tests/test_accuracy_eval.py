"""Accuracy evaluation framework for the Aegis moderation pipeline.

Target metrics (Phase 4):
  False negatives < 1 %   (harmful content that reaches users)
  False positives < 3 %   (legitimate cultural content blocked)

Test categories (1 000-image set):
  200  adult           Ã¢â‚¬â€ explicit content; all should be REJECTED
  200  safe            Ã¢â‚¬â€ clean photos, landscapes; all should be APPROVED
  100  violence        Ã¢â‚¬â€ gore, assault footage; should be REJECTED
  100  heritage        Ã¢â‚¬â€ temples, festivals, murtis; should be APPROVED
  100  festivals       Ã¢â‚¬â€ Holi, Diwali, processions; should be APPROVED
  100  children        Ã¢â‚¬â€ children at festivals; should be APPROVED
  100  scams           Ã¢â‚¬â€ phishing screenshots; should be REJECTED
  100  screenshots     Ã¢â‚¬â€ phone/app screenshots; typically APPROVED/UNDER_REVIEW
  100  ambiguous       Ã¢â‚¬â€ borderline content; outcome varies

Usage:
    python -m pytest tests/test_accuracy_eval.py -v                    # all unit tests
    python -m tests.test_accuracy_eval --images-dir /data/eval_set     # full eval
    python -m tests.test_accuracy_eval --category heritage --images-dir /data/eval_set
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

logger = logging.getLogger(__name__)

# Ã¢â€â‚¬Ã¢â€â‚¬ Decision engine unit tests Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

from backend.pipeline import decision_engine as engine  # noqa: E402


def _s(**kw: float) -> dict[str, float]:
    """Build a minimal scores dict with all keys defaulting to 0."""
    base = {
        k: 0.0
        for k in (
            "adult_score",
            "heritage_score",
            "content_quality_score",
            "child_safety_score",
            "child_presence_score",
            "violence_self_harm_score",
            "weapon_score",
            "blood_score",
            "self_harm_score",
            "promotion_score",
            "advertising_score",
            "affiliate_score",
            "social_media_score",
            "terrorism_score",
            "fraud_score",
            "hate_speech_score",
            "harassment_score",
            "misinformation_score",
            "self_harm_text_score",
            "pii_score",
            "llama_risk_score",
            "llama_approves",
            "ensemble_risk_score",
        )
    }
    base.update(kw)
    return base


class TestPhase4Thresholds(TestCase):
    """Verify all Phase 4 threshold values are set correctly."""

    def test_adult_reject_threshold(self) -> None:
        self.assertEqual(engine.ADULT_REJECT_THRESHOLD, 0.65)

    def test_heritage_review_threshold(self) -> None:
        self.assertEqual(engine.HERITAGE_REVIEW_THRESHOLD, 0.50)

    def test_heritage_exception_threshold(self) -> None:
        self.assertEqual(engine.HERITAGE_EXCEPTION_THRESHOLD, 0.45)

    def test_content_quality_threshold(self) -> None:
        self.assertEqual(engine.CONTENT_QUALITY_THRESHOLD, 0.35)

    def test_child_safety_threshold(self) -> None:
        self.assertEqual(engine.CHILD_SAFETY_THRESHOLD, 0.45)

    def test_child_safety_review_threshold(self) -> None:
        self.assertEqual(engine.CHILD_SAFETY_REVIEW_THRESHOLD, 0.65)

    def test_terrorism_threshold(self) -> None:
        self.assertEqual(engine.TERRORISM_THRESHOLD, 0.60)

    def test_drug_trafficking_threshold(self) -> None:
        self.assertEqual(engine.DRUG_TRAFFICKING_THRESHOLD, 0.60)

    def test_human_trafficking_threshold(self) -> None:
        self.assertEqual(engine.HUMAN_TRAFFICKING_THRESHOLD, 0.60)

    def test_violence_self_harm_threshold(self) -> None:
        self.assertEqual(engine.VIOLENCE_SELF_HARM_THRESHOLD, 0.65)

    def test_weapon_threshold(self) -> None:
        self.assertEqual(engine.WEAPON_THRESHOLD, 0.70)

    def test_blood_threshold(self) -> None:
        self.assertEqual(engine.BLOOD_THRESHOLD, 0.65)

    def test_self_harm_threshold(self) -> None:
        self.assertEqual(engine.SELF_HARM_THRESHOLD, 0.65)

    def test_privacy_threshold(self) -> None:
        self.assertEqual(engine.PRIVACY_THRESHOLD, 0.60)

    def test_fraud_threshold(self) -> None:
        self.assertEqual(engine.FRAUD_THRESHOLD, 0.60)

    def test_hate_speech_threshold(self) -> None:
        self.assertEqual(engine.HATE_SPEECH_THRESHOLD, 0.70)

    def test_harassment_threshold(self) -> None:
        self.assertEqual(engine.HARASSMENT_THRESHOLD, 0.65)

    def test_promotion_threshold(self) -> None:
        self.assertEqual(engine.PROMOTION_THRESHOLD, 0.70)

    def test_ensemble_risk_threshold(self) -> None:
        self.assertEqual(engine.ENSEMBLE_RISK_THRESHOLD, 0.75)

    def test_llama_rejection_confidence(self) -> None:
        self.assertEqual(engine.LLAMA_REJECTION_CONFIDENCE, 0.92)

    def test_llama_approval_confidence(self) -> None:
        self.assertEqual(engine.LLAMA_APPROVAL_CONFIDENCE, 0.95)


class TestDisagreementDetection(TestCase):
    """Tier 9 Ã¢â‚¬â€ sub-threshold risk + high heritage Ã¢â€ â€™ UNDER_REVIEW."""

    def test_weapon_plus_heritage_routes_to_review(self) -> None:
        # weapon_score=0.73; cultural protection Ã¢â€ â€™ weapon_eval=0.73*0.70=0.511
        # 0.50 < 0.511 < 0.90 AND heritage=0.90 > 0.50 Ã¢â€ â€™ disagree fires
        # weapon_eval=0.511 < WEAPON_THRESHOLD(0.70) Ã¢â€ â€™ weapon tier does NOT fire first
        scores = _s(weapon_score=0.73, heritage_score=0.90)
        decision, reason_code, _ = engine.decide_with_reason_code(scores)
        self.assertEqual(decision, "UNDER_REVIEW")
        self.assertEqual(reason_code, engine.DISAGREE_CONTENT)

    def test_violence_plus_heritage_routes_to_review(self) -> None:
        # violence_score=0.73; cultural protection Ã¢â€ â€™ violence_eval=0.73*0.70=0.511
        # 0.50 < 0.511 < 0.90 AND heritage=0.85 > 0.50 Ã¢â€ â€™ disagree fires
        # violence_eval=0.511 < VIOLENCE_SELF_HARM_THRESHOLD(0.65) Ã¢â€ â€™ violence tier does NOT fire
        scores = _s(violence_self_harm_score=0.73, heritage_score=0.85)
        decision, reason_code, _ = engine.decide_with_reason_code(scores)
        self.assertEqual(decision, "UNDER_REVIEW")
        self.assertEqual(reason_code, engine.DISAGREE_CONTENT)

    def test_child_plus_heritage_routes_to_review(self) -> None:
        # Phase 5 child safety dominance: child_eval = raw child score (NO 0.80 reduction).
        # child=0.55 > _DISAGREE_RISK_LOW(0.50) AND heritage=0.80 > _DISAGREE_HERITAGE(0.50)
        # Ã¢â€ â€™ disagreement detection fires Ã¢â€ â€™ UNDER_REVIEW DISAGREE_CONTENT
        scores = _s(child_safety_score=0.55, heritage_score=0.80)
        decision, reason_code, _ = engine.decide_with_reason_code(scores)
        self.assertEqual(decision, "UNDER_REVIEW")
        self.assertEqual(reason_code, engine.DISAGREE_CONTENT)

    def test_overwhelming_risk_suppresses_disagree(self) -> None:
        # weapon=0.95 (overwhelming, Ã¢â€°Â¥ 0.90 high-threshold) Ã¢â‚¬â€ should REJECT
        # No heritage Ã¢â€ â€™ no cultural protection Ã¢â€ â€™ weapon_eval = 0.95
        scores = _s(weapon_score=0.95)
        decision, reason_code, _ = engine.decide_with_reason_code(scores)
        self.assertEqual(decision, "REJECTED")
        self.assertEqual(reason_code, engine.WEAPON_CONTENT)

    def test_low_risk_no_disagree(self) -> None:
        # weapon=0.40 (below 0.50 disagree-low threshold) Ã¢â€ â€™ no disagree
        scores = _s(weapon_score=0.40, heritage_score=0.90)
        decision, _, _ = engine.decide_with_reason_code(scores)
        self.assertEqual(decision, "APPROVED")

    def test_heritage_without_risk_no_disagree(self) -> None:
        scores = _s(heritage_score=0.95)
        decision, _, _ = engine.decide_with_reason_code(scores)
        self.assertEqual(decision, "APPROVED")


class TestCulturalProtectionDecisions(TestCase):
    """Cultural protection: heritage_score > 0.60 reduces effective eval scores."""

    def test_high_weapon_score_reduced_by_heritage(self) -> None:
        # weapon_score raw = 0.68; after Ãƒâ€” 0.70 = 0.476 < WEAPON_THRESHOLD(0.70)
        # heritage_score = 0.90 > 0.60 Ã¢â€ â€™ protection applies
        scores = _s(weapon_score=0.68, heritage_score=0.90)
        decision, reason_code, _ = engine.decide_with_reason_code(scores)
        # weapon_eval = 0.68 * 0.70 = 0.476 < 0.70 Ã¢â€ â€™ weapon tier doesn't fire
        # disagree: max_non_adult = 0.476 < 0.50 Ã¢â€ â€™ no disagree Ã¢â€ â€™ APPROVED
        self.assertNotEqual(reason_code, engine.WEAPON_CONTENT)

    def test_violence_plus_heritage_routes_via_heritage_exception(self) -> None:
        # violence_score raw = 0.75 > VIOLENCE_SELF_HARM_THRESHOLD=0.65
        # With heritage 0.90: violence_eval = 0.75 * 0.70 = 0.525 < 0.65
        # No violence tier fires; also disagree: 0.525 > 0.50 and heritage=0.90 > 0.50
        # Ã¢â€ â€™ DISAGREE_CONTENT UNDER_REVIEW
        scores = _s(violence_self_harm_score=0.75, heritage_score=0.90)
        decision, reason_code, _ = engine.decide_with_reason_code(scores)
        self.assertNotEqual(decision, "REJECTED")
        self.assertNotEqual(reason_code, engine.VIOLENCE_CONTENT)

    def test_no_heritage_violence_is_rejected(self) -> None:
        # Same violence score, no heritage Ã¢â€ â€™ REJECTED
        scores = _s(violence_self_harm_score=0.75)
        decision, reason_code, _ = engine.decide_with_reason_code(scores)
        self.assertEqual(decision, "REJECTED")
        self.assertEqual(reason_code, engine.VIOLENCE_CONTENT)

    def test_heritage_does_not_protect_overwhelming_child_risk(self) -> None:
        # child_eval = 0.90 * 0.80 = 0.72 > CHILD_SAFETY_REVIEW_THRESHOLD(0.65)
        scores = _s(child_safety_score=0.90, heritage_score=0.95)
        decision, reason_code, _ = engine.decide_with_reason_code(scores)
        self.assertEqual(decision, "UNDER_REVIEW")
        self.assertEqual(reason_code, engine.CHILD_SAFETY_ALERT)


class TestEnsembleFormula(TestCase):
    """Validate the Phase 4 weighted ensemble formula."""

    def test_ensemble_weights_sum_correctly(self) -> None:
        from backend.pipeline.safety_flags import _compute_ensemble

        # All risk signals = 1.0, heritage = 0.0 Ã¢â€ â€™ max ensemble
        # 0.25 + 0.20 + 0.15 + 0.10 + 0.10 + (1-0)*0.05 + llama*0.15
        # With llama_result REJECTED conf=1.0: llama_risk=1.0
        # = 0.25 + 0.20 + 0.15 + 0.10 + 0.10 + 0.05 + 0.15 = 1.00
        result = _compute_ensemble(
            adult_score=1.0,
            child_score=1.0,
            violence_score=1.0,
            fraud_score=1.0,
            weapon_score=1.0,
            heritage_score=0.0,
            llama_result={"decision": "REJECTED", "confidence": 1.0},
        )
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_heritage_reduces_ensemble_risk(self) -> None:
        from backend.pipeline.safety_flags import _compute_ensemble

        # heritage=0 contributes (1-0)*0.05 = 0.05
        # heritage=1 contributes (1-1)*0.05 = 0.00  Ã¢â€ â€™ lower ensemble
        low_heritage = _compute_ensemble(
            adult_score=0.5,
            child_score=0.5,
            violence_score=0.5,
            fraud_score=0.5,
            weapon_score=0.5,
            heritage_score=0.0,
            llama_result=None,
        )
        high_heritage = _compute_ensemble(
            adult_score=0.5,
            child_score=0.5,
            violence_score=0.5,
            fraud_score=0.5,
            weapon_score=0.5,
            heritage_score=1.0,
            llama_result=None,
        )
        self.assertGreater(low_heritage, high_heritage)

    def test_zero_scores_give_low_ensemble(self) -> None:
        from backend.pipeline.safety_flags import _compute_ensemble

        # All zeros, no llama Ã¢â€ â€™ (1-0)*0.05 + 0.5*0.15 = 0.05 + 0.075 = 0.125
        # (llama=None Ã¢â€ â€™ llama_risk=0.5)
        result = _compute_ensemble(
            adult_score=0.0,
            child_score=0.0,
            violence_score=0.0,
            fraud_score=0.0,
            weapon_score=0.0,
            heritage_score=0.0,
            llama_result=None,
        )
        self.assertLess(result, 0.20)


class TestMultiCaptionGeneration(TestCase):
    """generate_captions() deduplicates and degrades gracefully."""

    def test_generate_captions_deduplicates(self) -> None:
        from backend.pipeline import vlm_engine

        with (
            patch.object(vlm_engine, "_get_blip") as mock_get_blip,
            patch("PIL.Image.open") as mock_pil_open,
        ):
            state = MagicMock()
            state.torch.cuda.is_available.return_value = False
            # All three decode calls return the same string Ã¢â€ â€™ dedup to 1
            state.processor.decode.return_value = "a temple"
            state.model.generate.return_value = MagicMock()
            mock_get_blip.return_value = state

            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=MagicMock())
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_pil_open.return_value = mock_ctx

            captions = vlm_engine.generate_captions("fake.jpg", n=3)

        # With dedup, only 1 unique caption
        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0], "a temple")

    def test_generate_captions_returns_empty_on_failure(self) -> None:
        from backend.pipeline import vlm_engine

        with patch.object(vlm_engine, "_get_blip", side_effect=RuntimeError("no gpu")):
            captions = vlm_engine.generate_captions("fake.jpg")
        self.assertEqual(captions, [])

    def test_generate_caption_backward_compat(self) -> None:
        from backend.pipeline import vlm_engine

        with patch.object(vlm_engine, "generate_captions", return_value=["a gopuram"]):
            cap = vlm_engine.generate_caption("fake.jpg")
        self.assertEqual(cap, "a gopuram")

    def test_generate_caption_returns_empty_string_on_empty_list(self) -> None:
        from backend.pipeline import vlm_engine

        with patch.object(vlm_engine, "generate_captions", return_value=[]):
            cap = vlm_engine.generate_caption("fake.jpg")
        self.assertEqual(cap, "")


class TestReasonModerationMultiCaption(TestCase):
    """reason_moderation() accepts blip_captions list and falls back gracefully."""

    def _patched_reason(self, **kwargs):
        from backend.pipeline import vlm_engine

        with patch.object(vlm_engine, "_get_llama", side_effect=RuntimeError("no gpu")):
            return vlm_engine.reason_moderation(**kwargs)

    def test_fallback_on_llama_failure(self) -> None:
        result = self._patched_reason(
            nsfw_score=0.1,
            objects_detected=[],
            ocr_text="",
            caption="temple",
            blip_captions=["a gopuram", "a temple tower"],
        )
        self.assertEqual(result["decision"], "UNDER_REVIEW")

    def test_accepts_legacy_blip_caption_param(self) -> None:
        result = self._patched_reason(
            nsfw_score=0.1,
            objects_detected=[],
            ocr_text="",
            caption="temple",
            blip_caption="a gopuram",  # legacy single-caption param
        )
        self.assertEqual(result["decision"], "UNDER_REVIEW")

    def test_accepts_blip2_caption_backward_compat(self) -> None:
        result = self._patched_reason(
            nsfw_score=0.1,
            objects_detected=[],
            ocr_text="",
            caption="temple",
            blip2_caption="a gopuram",  # oldest alias
        )
        self.assertEqual(result["decision"], "UNDER_REVIEW")


# Ã¢â€â‚¬Ã¢â€â‚¬ Full evaluation harness (runs against real images) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


@dataclass
class EvalResult:
    category: str
    total: int
    correct: int
    false_positives: int  # predicted REJECTED/REVIEW when ground truth = APPROVED
    false_negatives: int  # predicted APPROVED when ground truth = REJECTED
    errors: int
    details: list[dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        tp = self.correct - (
            self.total - self.correct - self.false_positives - self.false_negatives - self.errors
        )
        denom = tp + self.false_positives
        return tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        tp = self.correct - (
            self.total - self.correct - self.false_positives - self.false_negatives - self.errors
        )
        denom = tp + self.false_negatives
        return tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def fp_rate(self) -> float:
        denom = self.total - self.false_negatives - self.errors
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def fn_rate(self) -> float:
        denom = self.total - self.false_positives - self.errors
        return self.false_negatives / denom if denom > 0 else 0.0


# Category Ã¢â€ â€™ expected decision
CATEGORY_EXPECTED: dict[str, str] = {
    "adult": "REJECTED",
    "safe": "APPROVED",
    "violence": "REJECTED",
    "heritage": "APPROVED",
    "festivals": "APPROVED",
    "children": "APPROVED",
    "scams": "REJECTED",
    # screenshots and ambiguous have no hard expectation Ã¢â‚¬â€ treated as informational
    "screenshots": None,
    "ambiguous": None,
}


def _is_correct(prediction: str, expected: str | None) -> bool:
    if expected is None:
        return True  # informational category Ã¢â‚¬â€ all outcomes OK
    if expected == "APPROVED":
        return prediction == "APPROVED"
    if expected == "REJECTED":
        return prediction in ("REJECTED", "UNDER_REVIEW")
    return False


def evaluate_category(
    images_dir: Path,
    category: str,
    caption_fn: Callable[[Path], str] | None = None,
) -> EvalResult:
    """Evaluate all images in images_dir/category/ against the expected outcome.

    Loads the real moderation pipeline Ã¢â‚¬â€ requires GPU + models to be available.
    caption_fn(path) optionally provides user-supplied captions.
    """
    from backend.pipeline.decision_engine import decide
    from backend.pipeline.safety_flags import analyze_image

    cat_dir = images_dir / category
    if not cat_dir.is_dir():
        logger.warning("Category dir not found: %s", cat_dir)
        return EvalResult(
            category=category, total=0, correct=0, false_positives=0, false_negatives=0, errors=0
        )

    expected = CATEGORY_EXPECTED.get(category)
    image_paths = sorted(
        p for p in cat_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    result = EvalResult(
        category=category,
        total=len(image_paths),
        correct=0,
        false_positives=0,
        false_negatives=0,
        errors=0,
    )

    for img_path in image_paths:
        caption = caption_fn(img_path) if caption_fn else None
        try:
            pipeline_result = analyze_image(str(img_path), caption)
            if pipeline_result.pipeline_error:
                result.errors += 1
                result.details.append({"path": str(img_path), "outcome": "ERROR"})
                continue

            decision, _ = decide(pipeline_result.scores)
            correct = _is_correct(decision, expected)

            if correct:
                result.correct += 1
            elif expected == "APPROVED" and decision != "APPROVED":
                result.false_positives += 1
            elif expected == "REJECTED" and decision == "APPROVED":
                result.false_negatives += 1

            result.details.append(
                {
                    "path": str(img_path),
                    "decision": decision,
                    "expected": expected,
                    "correct": correct,
                    "adult_score": pipeline_result.scores.get("adult_score"),
                    "heritage_score": pipeline_result.scores.get("heritage_score"),
                    "ensemble_risk": pipeline_result.scores.get("ensemble_risk_score"),
                }
            )

        except Exception as exc:
            logger.exception("Error processing %s", img_path)
            result.errors += 1
            result.details.append(
                {"path": str(img_path), "outcome": "EXCEPTION", "error": str(exc)}
            )

    return result


def evaluate_all(images_dir: Path) -> dict[str, EvalResult]:
    """Run evaluation for all 9 categories."""
    return {cat: evaluate_category(images_dir, cat) for cat in CATEGORY_EXPECTED}


def print_report(results: dict[str, EvalResult]) -> None:
    total_fp = sum(r.false_positives for r in results.values())
    total_fn = sum(r.false_negatives for r in results.values())
    total_images = sum(r.total for r in results.values())
    total_errors = sum(r.errors for r in results.values())

    print("\n" + "=" * 70)
    print(f"MODERATION ACCURACY REPORT ({total_images} images, {total_errors} errors)")
    print("=" * 70)
    header = f"{'Category':<14} {'Total':>6} {'FP':>5} {'FN':>5} {'FP%':>7} {'FN%':>7} {'F1':>7}"
    print(header)
    print("-" * 70)
    for cat, r in results.items():
        if r.total == 0:
            continue
        print(
            f"{cat:<14} {r.total:>6} {r.false_positives:>5} {r.false_negatives:>5}"
            f" {r.fp_rate*100:>6.1f}% {r.fn_rate*100:>6.1f}% {r.f1:>6.3f}"
        )
    print("-" * 70)
    overall_fn_rate = total_fn / max(1, total_images - total_fp - total_errors) * 100
    overall_fp_rate = total_fp / max(1, total_images - total_fn - total_errors) * 100
    print(
        f"{'OVERALL':<14} {total_images:>6} {total_fp:>5} {total_fn:>5}"
        f" {overall_fp_rate:>6.1f}% {overall_fn_rate:>6.1f}%"
    )
    print("=" * 70)
    target_fp_ok = overall_fp_rate <= 3.0
    target_fn_ok = overall_fn_rate <= 1.0
    print(f"FP target (Ã¢â€°Â¤ 3.0%): {'Ã¢Å“â€œ PASS' if target_fp_ok else 'Ã¢Å“â€” FAIL'}")
    print(f"FN target (Ã¢â€°Â¤ 1.0%): {'Ã¢Å“â€œ PASS' if target_fn_ok else 'Ã¢Å“â€” FAIL'}")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Accuracy evaluation for moderation pipeline")
    parser.add_argument("--images-dir", required=True, type=Path, help="Root eval images directory")
    parser.add_argument("--category", default=None, help="Evaluate a single category")
    parser.add_argument("--output", default=None, help="Write full JSON results to file")
    args = parser.parse_args()

    if args.category:
        result = evaluate_category(args.images_dir, args.category)
        print_report({args.category: result})
        if args.output:
            Path(args.output).write_text(json.dumps({args.category: result.details}, indent=2))
    else:
        results = evaluate_all(args.images_dir)
        print_report(results)
        if args.output:
            Path(args.output).write_text(
                json.dumps({cat: r.details for cat, r in results.items()}, indent=2)
            )
