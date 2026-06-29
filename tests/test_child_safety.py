"""Unit tests for child safety signal fusion."""

from unittest import TestCase

from pipeline.child_safety import analyze_child_safety, get_child_safety_scores


class ChildSafetyTests(TestCase):
    def test_empty_inputs_return_zero_scores(self) -> None:
        result = analyze_child_safety([], {})

        self.assertEqual(
            result,
            {
                "child_presence_score": 0.0,
                "child_safety_score": 0.0,
            },
        )

    def test_explicit_yolo_child_label_produces_conservative_scores(self) -> None:
        result = analyze_child_safety(
            [{"class": "toddler", "confidence": 0.9}],
            {},
        )

        self.assertEqual(result["child_presence_score"], 0.9)
        self.assertGreater(result["child_safety_score"], 0.0)
        self.assertLess(result["child_safety_score"], result["child_presence_score"])

    def test_yolo_person_alone_is_only_weak_presence_not_safety(self) -> None:
        result = analyze_child_safety(
            [{"class": "person", "confidence": 0.96}],
            {},
        )

        self.assertAlmostEqual(result["child_presence_score"], 0.24)
        self.assertEqual(result["child_safety_score"], 0.0)

    def test_person_detection_and_clip_child_signal_are_combined(self) -> None:
        result = analyze_child_safety(
            [{"class": "person", "confidence": 0.95}],
            {"school child": 0.72},
        )

        self.assertEqual(result["child_presence_score"], 0.72)
        self.assertGreater(result["child_safety_score"], 0.4)
        self.assertLessEqual(result["child_safety_score"], 1.0)

    def test_nested_clip_score_mapping_is_supported(self) -> None:
        result = analyze_child_safety(
            [{"class": "car", "confidence": 0.88}],
            {"prompt_scores": {"infant": 0.64, "adult": 0.9}},
        )

        self.assertEqual(result["child_presence_score"], 0.64)
        self.assertGreater(result["child_safety_score"], 0.0)

    def test_non_child_clip_labels_are_ignored(self) -> None:
        result = analyze_child_safety(
            [{"class": "car", "confidence": 0.88}],
            {"adult": 0.99, "vehicle": 0.8},
        )

        self.assertEqual(result["child_presence_score"], 0.0)
        self.assertEqual(result["child_safety_score"], 0.0)

    def test_compatibility_alias_matches_primary_api(self) -> None:
        detections = [{"class": "minor", "confidence": 0.7}]
        clip_embeddings = {"minor": 0.6}

        self.assertEqual(
            get_child_safety_scores(detections, clip_embeddings),
            analyze_child_safety(detections, clip_embeddings),
        )
