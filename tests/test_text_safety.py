"""Unit tests for advanced rule-based text safety signals."""

from unittest import TestCase

from backend.pipeline.text_safety import analyze_text_safety, get_text_safety_scores


EXPECTED_KEYS = {
    "terrorism_score",
    "fraud_score",
    "hate_speech_score",
    "harassment_score",
    "misinformation_score",
    "self_harm_text_score",
}


class TextSafetyTests(TestCase):
    def test_empty_inputs_return_zero_scores(self) -> None:
        result = analyze_text_safety("", None)

        self.assertEqual(set(result), EXPECTED_KEYS)
        self.assertTrue(all(score == 0.0 for score in result.values()))

    def test_clean_heritage_text_scores_zero(self) -> None:
        result = analyze_text_safety(
            "Brihadeeswarar temple inscription",
            "Ancient Chola heritage",
        )

        self.assertTrue(all(score == 0.0 for score in result.values()))

    def test_illegal_content_rules_detect_terrorism_and_weapon_instructions(self) -> None:
        result = analyze_text_safety(
            "Extremist propaganda and bomb making instructions",
            "weapon construction instructions",
        )

        self.assertGreater(result["terrorism_score"], 0.5)
        self.assertEqual(result["fraud_score"], 0.0)

    def test_drug_and_human_trafficking_raise_terrorism_bucket(self) -> None:
        result = analyze_text_safety(
            "Drug trafficking network",
            "human trafficking victims",
        )

        self.assertGreater(result["terrorism_score"], 0.0)

    def test_fraud_rules_detect_phishing_and_fake_investments(self) -> None:
        result = analyze_text_safety(
            "Phishing alert verify your password",
            "fake investment scheme with guaranteed income",
        )

        self.assertGreater(result["fraud_score"], 0.5)
        self.assertEqual(result["hate_speech_score"], 0.0)

    def test_hate_speech_rules_detect_protected_class_attacks(self) -> None:
        result = analyze_text_safety(
            "religion based attack and caste based attack",
            "ethnicity based attack",
        )

        self.assertGreater(result["hate_speech_score"], 0.5)
        self.assertEqual(result["fraud_score"], 0.0)

    def test_harassment_rules_detect_bullying_and_threats(self) -> None:
        result = analyze_text_safety(
            "Targeted abuse and bullying",
            "I will hurt you",
        )

        self.assertGreater(result["harassment_score"], 0.5)

    def test_self_harm_text_rules_detect_encouragement(self) -> None:
        result = analyze_text_safety(
            "suicide encouragement",
            "end your life",
        )

        self.assertGreater(result["self_harm_text_score"], 0.5)
        self.assertEqual(result["harassment_score"], 0.0)

    def test_misinformation_rules_detect_medical_and_election_misinformation(self) -> None:
        result = analyze_text_safety(
            "medical misinformation vaccines cause infertility",
            "election manipulation wrong voting date",
        )

        self.assertGreater(result["misinformation_score"], 0.5)
        self.assertEqual(result["terrorism_score"], 0.0)

    def test_alias_matches_primary_api(self) -> None:
        ocr_text = "fake government notice"
        caption = "official government order"

        self.assertEqual(
            get_text_safety_scores(ocr_text, caption),
            analyze_text_safety(ocr_text, caption),
        )
