"""Unit tests for privacy and PII detection."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from pipeline import pii_detector
from pipeline.pii_detector import analyze_pii, get_pii_scores


class PiiDetectorTests(TestCase):
    def setUp(self) -> None:
        pii_detector._get_presidio_analyzer.cache_clear()

    def tearDown(self) -> None:
        pii_detector._get_presidio_analyzer.cache_clear()

    def test_empty_inputs_return_zero_and_false_flags(self) -> None:
        result = analyze_pii("", None)

        self.assertEqual(result["pii_score"], 0.0)
        self.assertFalse(result["aadhaar_detected"])
        self.assertFalse(result["pan_detected"])
        self.assertFalse(result["passport_detected"])
        self.assertFalse(result["phone_detected"])
        self.assertFalse(result["email_detected"])
        self.assertFalse(result["bank_info_detected"])

    def test_phone_and_email_are_detected(self) -> None:
        result = analyze_pii(
            "Contact +91 98765 43210",
            "email user@example.com",
        )

        self.assertTrue(result["phone_detected"])
        self.assertTrue(result["email_detected"])
        self.assertGreater(result["pii_score"], 0.0)

    def test_aadhaar_number_is_detected(self) -> None:
        result = analyze_pii("Aadhaar 2345 6789 0123", None)

        self.assertTrue(result["aadhaar_detected"])
        self.assertGreaterEqual(result["pii_score"], 0.55)

    def test_pan_number_is_detected(self) -> None:
        result = analyze_pii("PAN ABCDE1234F", None)

        self.assertTrue(result["pan_detected"])
        self.assertGreater(result["pii_score"], 0.0)

    def test_passport_number_is_detected(self) -> None:
        result = analyze_pii("Passport Z1234567", None)

        self.assertTrue(result["passport_detected"])
        self.assertGreater(result["pii_score"], 0.0)

    def test_bank_account_upi_and_ifsc_are_detected(self) -> None:
        result = analyze_pii(
            "Account 123456789012 IFSC HDFC0123456",
            "UPI user@oksbi",
        )

        self.assertTrue(result["bank_info_detected"])
        self.assertGreaterEqual(result["pii_score"], 0.5)

    def test_personal_address_contributes_to_pii_score(self) -> None:
        result = analyze_pii(
            "House 42 MG Road",
            "Pincode 560001",
        )

        self.assertGreater(result["pii_score"], 0.0)
        self.assertFalse(result["bank_info_detected"])

    def test_clean_text_scores_zero(self) -> None:
        result = analyze_pii(
            "Brihadeeswarar temple inscription",
            "Ancient Chola heritage",
        )

        self.assertEqual(result["pii_score"], 0.0)
        self.assertFalse(result["phone_detected"])
        self.assertFalse(result["email_detected"])

    def test_presidio_results_are_used_when_available(self) -> None:
        analyzer = Mock()
        analyzer.analyze.return_value = [
            SimpleNamespace(entity_type="EMAIL_ADDRESS"),
            SimpleNamespace(entity_type="PHONE_NUMBER"),
        ]

        with patch("pipeline.pii_detector._get_presidio_analyzer", return_value=analyzer):
            result = analyze_pii("contact details", None)

        self.assertTrue(result["email_detected"])
        self.assertTrue(result["phone_detected"])

    def test_presidio_failure_falls_back_to_regex(self) -> None:
        analyzer = Mock()
        analyzer.analyze.side_effect = RuntimeError("presidio failed")

        with patch("pipeline.pii_detector._get_presidio_analyzer", return_value=analyzer):
            result = analyze_pii("email user@example.com", None)

        self.assertTrue(result["email_detected"])
        self.assertFalse(result["phone_detected"])

    def test_alias_matches_primary_api(self) -> None:
        ocr_text = "PAN ABCDE1234F"
        caption = "phone 9876543210"

        self.assertEqual(get_pii_scores(ocr_text, caption), analyze_pii(ocr_text, caption))
