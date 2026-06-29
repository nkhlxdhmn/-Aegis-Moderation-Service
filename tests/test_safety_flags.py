"""Unit tests for the Phase 4 moderation pipeline orchestration."""

from unittest import TestCase
from unittest.mock import patch

from backend.pipeline.decision_engine import decide
from backend.pipeline.clip_engine import ClipAnalysisResult
from backend.pipeline.safety_flags import analyze_image


EXPECTED_SCORE_KEYS = {
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
    # Ensemble + VLM signals added in production upgrade
    "llama_risk_score",
    "llama_approves",
    "ensemble_risk_score",
    # Phase 5: uncertainty estimation
    "uncertainty_score",
}

_LLAMA_APPROVED = {
    "decision": "APPROVED",
    "reason": "Cultural heritage content.",
    "confidence": 0.5,
    "category": "Heritage",
}
_LLAMA_FALLBACK = {
    "decision": "UNDER_REVIEW",
    "reason": "Reasoning unavailable.",
    "confidence": 0.5,
    "category": "Uncategorized",
}


class SafetyFlagsTests(TestCase):
    def test_nudenet_failure_marks_pipeline_error(self) -> None:
        with patch("backend.pipeline.safety_flags.Path.is_file", return_value=True), \
             patch("backend.pipeline.safety_flags.image_quality.check_image_quality", return_value=(True, None)), \
             patch("backend.pipeline.safety_flags.hash_cache.lookup", return_value=None), \
             patch("backend.pipeline.safety_flags.nsfw.get_adult_score", side_effect=RuntimeError("boom")):
            result = analyze_image("image.jpg", "caption")

        self.assertTrue(result.pipeline_error)
        self.assertIn("adult_score", result.scores)
        self.assertEqual(result.category_scores, {})

    def test_openclip_failure_marks_pipeline_error(self) -> None:
        with patch("backend.pipeline.safety_flags.Path.is_file", return_value=True), \
             patch("backend.pipeline.safety_flags.image_quality.check_image_quality", return_value=(True, None)), \
             patch("backend.pipeline.safety_flags.hash_cache.lookup", return_value=None), \
             patch("backend.pipeline.safety_flags.nsfw.get_adult_score", return_value=0.8), \
             patch("backend.pipeline.safety_flags.ocr.extract_ocr_text", return_value=""), \
             patch("backend.pipeline.safety_flags.ocr.get_text_quality_score", return_value=0.0), \
             patch("backend.pipeline.safety_flags.clip_engine.analyze_content", side_effect=RuntimeError("clip failed")):
            result = analyze_image("image.jpg", "caption")

        self.assertTrue(result.pipeline_error)
        self.assertEqual(result.scores["adult_score"], 0.8)

    def test_ocr_failure_continues_with_empty_text(self) -> None:
        with patch("backend.pipeline.safety_flags.Path.is_file", return_value=True), \
             patch("backend.pipeline.safety_flags.image_quality.check_image_quality", return_value=(True, None)), \
             patch("backend.pipeline.safety_flags.hash_cache.lookup", return_value=None), \
             patch("backend.pipeline.safety_flags.hash_cache.store", return_value=None), \
             patch("backend.pipeline.safety_flags.nsfw.get_adult_score", return_value=0.05), \
             patch("backend.pipeline.safety_flags.ocr.extract_ocr_text", return_value=""), \
             patch("backend.pipeline.safety_flags.ocr.get_text_quality_score", return_value=0.0), \
             patch("backend.pipeline.safety_flags.clip_engine.analyze_content", return_value=ClipAnalysisResult(
                category_scores={"Religious & Spiritual Heritage": 0.9},
                heritage_score=0.9,
             )), \
             patch("backend.pipeline.safety_flags._detect_objects", return_value=[]), \
             patch("backend.pipeline.safety_flags.vlm_engine.generate_captions", return_value=[""]), \
             patch(
            "backend.pipeline.safety_flags.vlm_engine.reason_moderation",
            return_value=_LLAMA_APPROVED,
        ):
            result = analyze_image("image.jpg", "caption")

        self.assertFalse(result.pipeline_error)
        self.assertEqual(result.ocr_text, "")
        self.assertEqual(result.scores["content_quality_score"], 0.0)

    def test_score_schema_is_decision_engine_compatible(self) -> None:
        with patch("backend.pipeline.safety_flags.Path.is_file", return_value=True), \
             patch("backend.pipeline.safety_flags.image_quality.check_image_quality", return_value=(True, None)), \
             patch("backend.pipeline.safety_flags.hash_cache.lookup", return_value=None), \
             patch("backend.pipeline.safety_flags.hash_cache.store", return_value=None), \
             patch("backend.pipeline.safety_flags.nsfw.get_adult_score", return_value=0.05), \
             patch("backend.pipeline.safety_flags.ocr.extract_ocr_text", return_value="text"), \
             patch("backend.pipeline.safety_flags.ocr.get_text_quality_score", return_value=0.1), \
             patch("backend.pipeline.safety_flags.clip_engine.analyze_content", return_value=ClipAnalysisResult(
                category_scores={"Religious & Spiritual Heritage": 0.9},
                heritage_score=0.9,
             )), \
             patch("backend.pipeline.safety_flags._detect_objects", return_value=[]), \
             patch("backend.pipeline.safety_flags.vlm_engine.generate_captions", return_value=[""]), \
             patch("backend.pipeline.safety_flags.vlm_engine.reason_moderation", return_value=_LLAMA_APPROVED):
            result = analyze_image("image.jpg", "caption")

        self.assertEqual(set(result.scores), EXPECTED_SCORE_KEYS)
        decision, _ = decide(result.scores)
        self.assertEqual(decision, "APPROVED")

    def test_pipeline_uses_single_openclip_analysis_call(self) -> None:
        clip_result = ClipAnalysisResult(
            category_scores={"Religious & Spiritual Heritage": 0.9},
            heritage_score=0.9,
        )

        with patch("backend.pipeline.safety_flags.Path.is_file", return_value=True), \
             patch("backend.pipeline.safety_flags.image_quality.check_image_quality", return_value=(True, None)), \
             patch("backend.pipeline.safety_flags.hash_cache.lookup", return_value=None), \
             patch("backend.pipeline.safety_flags.hash_cache.store", return_value=None), \
             patch("backend.pipeline.safety_flags.nsfw.get_adult_score", return_value=0.05), \
             patch("backend.pipeline.safety_flags.text_detector.detect_text_regions", return_value=(True, {})), \
             patch("backend.pipeline.safety_flags.ocr.extract_ocr_text", return_value="text"), \
             patch("backend.pipeline.safety_flags.ocr.get_text_quality_score", return_value=0.1), \
             patch("backend.pipeline.safety_flags.clip_engine.analyze_content", return_value=clip_result) as analyze_content, \
             patch("backend.pipeline.safety_flags.clip_engine.get_image_embedding", return_value=None), \
             patch("backend.pipeline.safety_flags.embedding_cache.search_similar_image", return_value=None), \
             patch("backend.pipeline.safety_flags.embedding_cache.store_image", return_value=True), \
             patch("backend.pipeline.safety_flags._detect_objects", return_value=[]), \
             patch("backend.pipeline.safety_flags.clip_engine.get_category_scores") as get_category_scores, \
             patch("backend.pipeline.safety_flags.clip_engine.get_heritage_score") as get_heritage_score, \
             patch("backend.pipeline.safety_flags.qwen_vl.describe_image", return_value={"description": "", "confidence": 0.5}), \
             patch("backend.pipeline.safety_flags.vlm_engine.generate_captions", return_value=[""]), \
             patch("backend.pipeline.safety_flags.vlm_engine.reason_moderation", return_value=_LLAMA_APPROVED):
            result = analyze_image("image.jpg", "caption")

        self.assertFalse(result.pipeline_error)
        # Smart OCR was forced on via text_detector patch, so "text" is passed to SigLIP.
        analyze_content.assert_called_once_with("image.jpg", "caption", "text")
        get_category_scores.assert_not_called()
        get_heritage_score.assert_not_called()

    def test_phase2_signal_scores_are_merged_into_score_schema(self) -> None:
        clip_result = ClipAnalysisResult(
            category_scores={"Religious & Spiritual Heritage": 0.9},
            heritage_score=0.9,
            safety_scores={"weapon": 0.7},
            child_scores={"child": 0.8},
            promotion_scores={"sponsored ad": 0.6},
        )
        yolo_detections = [{"class": "person", "confidence": 0.95}]

        with patch("backend.pipeline.safety_flags.Path.is_file", return_value=True), \
             patch("backend.pipeline.safety_flags.image_quality.check_image_quality", return_value=(True, None)), \
             patch("backend.pipeline.safety_flags.hash_cache.lookup", return_value=None), \
             patch("backend.pipeline.safety_flags.hash_cache.store", return_value=None), \
             patch("backend.pipeline.safety_flags.nsfw.get_adult_score", return_value=0.05), \
             patch("backend.pipeline.safety_flags.ocr.extract_ocr_text", return_value="email user@example.com"), \
             patch("backend.pipeline.safety_flags.ocr.get_text_quality_score", return_value=0.1), \
             patch("backend.pipeline.safety_flags.clip_engine.analyze_content", return_value=clip_result), \
             patch("backend.pipeline.safety_flags._detect_objects", return_value=yolo_detections), \
             patch(
            "backend.pipeline.safety_flags.child_safety.analyze_child_safety",
            return_value={"child_presence_score": 0.8, "child_safety_score": 0.7},
        ) as child_analysis, patch(
            "backend.pipeline.safety_flags.safety_detector.analyze_safety",
            return_value={
                "weapon_score": 0.6,
                "blood_score": 0.0,
                "self_harm_score": 0.0,
                "violence_self_harm_score": 0.6,
            },
        ) as safety_analysis, patch(
            "backend.pipeline.safety_flags.promotion_detector.analyze_promotion",
            return_value={
                "promotion_score": 0.5,
                "advertising_score": 0.5,
                "affiliate_score": 0.0,
                "social_media_score": 0.0,
            },
        ) as promotion_analysis, patch(
            "backend.pipeline.safety_flags.text_safety.analyze_text_safety",
            return_value={
                "terrorism_score": 0.0,
                "fraud_score": 0.4,
                "hate_speech_score": 0.0,
                "harassment_score": 0.0,
                "misinformation_score": 0.0,
                "self_harm_text_score": 0.0,
            },
        ) as text_analysis, patch(
            "backend.pipeline.safety_flags.pii_detector.analyze_pii",
            return_value={
                "pii_score": 0.25,
                "aadhaar_detected": False,
                "pan_detected": False,
                "passport_detected": False,
                "phone_detected": False,
                "email_detected": True,
                "bank_info_detected": False,
            },
        ) as pii_analysis, patch(
            "backend.pipeline.safety_flags.vlm_engine.generate_captions",
            return_value=[""],
        ), patch(
            "backend.pipeline.safety_flags.vlm_engine.reason_moderation",
            return_value=_LLAMA_FALLBACK,
        ):
            result = analyze_image("image.jpg", "caption")

        self.assertFalse(result.pipeline_error)
        self.assertEqual(set(result.scores), EXPECTED_SCORE_KEYS)
        # Scores after cultural protection (heritage=0.9 > 0.60):
        # Phase 5 child safety dominance: child scores are NEVER reduced.
        # child_presence_score = 0.80 (raw, unchanged)
        # child_safety_score   = 0.70 (raw, unchanged)
        # weapon_score Ã— 0.70 = 0.42, violence_self_harm_score Ã— 0.70 = 0.42
        self.assertAlmostEqual(result.scores["child_presence_score"], 0.80, places=5)
        self.assertAlmostEqual(result.scores["child_safety_score"], 0.70, places=5)
        self.assertAlmostEqual(result.scores["weapon_score"], 0.42, places=5)
        self.assertAlmostEqual(result.scores["violence_self_harm_score"], 0.42, places=5)
        self.assertEqual(result.scores["promotion_score"], 0.5)
        self.assertEqual(result.scores["fraud_score"], 0.4)
        self.assertEqual(result.scores["pii_score"], 0.25)
        child_analysis.assert_called_once_with(yolo_detections, clip_result.child_scores)
        safety_analysis.assert_called_once_with(yolo_detections, clip_result.safety_scores)
        promotion_analysis.assert_called_once_with(
            "email user@example.com",
            "caption",
            clip_result.promotion_scores,
            yolo_detections,
        )
        text_analysis.assert_called_once_with("email user@example.com", "caption")
        pii_analysis.assert_called_once_with("email user@example.com", "caption")
