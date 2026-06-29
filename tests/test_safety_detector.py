"""Unit tests for violence and self-harm safety signal fusion."""

from unittest import TestCase

from pipeline.safety_detector import analyze_safety, get_safety_scores


class SafetyDetectorTests(TestCase):
    def test_empty_inputs_return_zero_scores(self) -> None:
        result = analyze_safety([], {})

        self.assertEqual(
            result,
            {
                "weapon_score": 0.0,
                "blood_score": 0.0,
                "self_harm_score": 0.0,
                "violence_self_harm_score": 0.0,
            },
        )

    def test_yolo_weapon_detection_contributes_to_weapon_score(self) -> None:
        result = analyze_safety(
            [{"class": "knife", "confidence": 0.8}],
            {},
        )

        self.assertGreater(result["weapon_score"], 0.0)
        self.assertEqual(result["blood_score"], 0.0)
        self.assertEqual(result["self_harm_score"], 0.0)
        self.assertEqual(
            result["violence_self_harm_score"],
            result["weapon_score"],
        )

    def test_clip_blood_signal_contributes_to_blood_score(self) -> None:
        result = analyze_safety(
            [{"class": "person", "confidence": 0.95}],
            {"blood": 0.7},
        )

        self.assertEqual(result["weapon_score"], 0.0)
        self.assertGreater(result["blood_score"], 0.0)
        self.assertEqual(
            result["violence_self_harm_score"],
            result["blood_score"],
        )

    def test_yolo_and_clip_signals_are_combined_conservatively(self) -> None:
        result = analyze_safety(
            [{"class": "gun", "confidence": 0.6}],
            {"weapon": 0.5},
        )

        # Combined score should exceed either individual signal (agreement bonus),
        # but not balloon to 1.0 the way the old 1-(1-a)(1-b) formula did.
        # Formula: max(yolo, clip) * 0.90 + geometric_mean * 0.15
        #        = 0.6 * 0.90 + sqrt(0.6*0.5) * 0.15 ≈ 0.622
        self.assertGreater(result["weapon_score"], 0.5)
        self.assertLessEqual(result["weapon_score"], 1.0)

    def test_self_harm_and_suicide_prompts_map_to_self_harm_score(self) -> None:
        result = analyze_safety(
            [],
            {"prompt_scores": {"self harm": 0.66, "suicide": 0.5}},
        )

        self.assertGreater(result["self_harm_score"], 0.0)
        self.assertEqual(
            result["violence_self_harm_score"],
            result["self_harm_score"],
        )

    def test_nested_safety_prompt_mapping_is_supported(self) -> None:
        result = analyze_safety(
            [{"class": "car", "confidence": 0.88}],
            {"safety_scores": {"dead body": 0.64, "temple": 0.9}},
        )

        self.assertGreater(result["blood_score"], 0.0)
        self.assertEqual(result["weapon_score"], 0.0)

    def test_general_violence_increases_aggregate_only(self) -> None:
        result = analyze_safety(
            [{"class": "violence", "confidence": 0.7}],
            {},
        )

        self.assertEqual(result["weapon_score"], 0.0)
        self.assertEqual(result["blood_score"], 0.0)
        self.assertEqual(result["self_harm_score"], 0.0)
        self.assertGreater(result["violence_self_harm_score"], 0.0)

    def test_unrelated_labels_are_ignored(self) -> None:
        result = analyze_safety(
            [{"class": "car", "confidence": 0.99}],
            {"festival": 0.9, "temple": 0.8},
        )

        self.assertEqual(result["weapon_score"], 0.0)
        self.assertEqual(result["blood_score"], 0.0)
        self.assertEqual(result["self_harm_score"], 0.0)
        self.assertEqual(result["violence_self_harm_score"], 0.0)

    def test_compatibility_alias_matches_primary_api(self) -> None:
        detections = [{"class": "rifle", "confidence": 0.75}]
        clip_safety_prompts = {"gun": 0.6}

        self.assertEqual(
            get_safety_scores(detections, clip_safety_prompts),
            analyze_safety(detections, clip_safety_prompts),
        )
