"""Unit tests for promotion signal fusion."""

from unittest import TestCase

from pipeline.promotion_detector import analyze_promotion, get_promotion_scores


class PromotionDetectorTests(TestCase):
    def test_empty_inputs_return_zero_scores(self) -> None:
        result = analyze_promotion("", None, {}, [])

        self.assertEqual(
            result,
            {
                "promotion_score": 0.0,
                "advertising_score": 0.0,
                "affiliate_score": 0.0,
                "social_media_score": 0.0,
            },
        )

    def test_ocr_and_caption_advertising_phrases_are_detected(self) -> None:
        result = analyze_promotion(
            "Sponsored ad limited time offer",
            "Buy now with promo code",
            {},
            [],
        )

        self.assertGreater(result["advertising_score"], 0.5)
        self.assertEqual(result["promotion_score"], result["advertising_score"])

    def test_affiliate_and_referral_phrases_are_detected(self) -> None:
        result = analyze_promotion(
            "",
            "Use my affiliate link and referral code",
            {},
            [],
        )

        self.assertGreater(result["affiliate_score"], 0.4)
        self.assertEqual(result["advertising_score"], 0.0)

    def test_social_media_promotions_are_detected(self) -> None:
        result = analyze_promotion(
            "Follow us and like and share",
            "Subscribe now to my YouTube channel",
            {},
            [],
        )

        self.assertGreater(result["social_media_score"], 0.5)
        self.assertEqual(result["promotion_score"], result["social_media_score"])

    def test_openclip_semantic_scores_are_detected(self) -> None:
        result = analyze_promotion(
            "",
            "",
            {"affiliate marketing": 0.68, "temple": 0.9},
            [],
        )

        self.assertEqual(result["affiliate_score"], 0.68)
        self.assertEqual(result["promotion_score"], 0.68)

    def test_nested_openclip_mapping_is_supported(self) -> None:
        result = analyze_promotion(
            "",
            "",
            {"promotion_scores": {"telegram promotion": 0.72}},
            [],
        )

        self.assertEqual(result["social_media_score"], 0.72)

    def test_yolo_product_detection_adds_weak_advertising_support(self) -> None:
        result = analyze_promotion(
            "",
            "",
            {},
            [{"class": "bottle", "confidence": 0.8}],
        )

        self.assertAlmostEqual(result["advertising_score"], 0.2)
        self.assertEqual(result["promotion_score"], result["advertising_score"])

    def test_urls_and_many_hashtags_add_conservative_text_support(self) -> None:
        result = analyze_promotion(
            "Visit https://example.com",
            "#one #two #three #four #five #six",
            {},
            [],
        )

        self.assertGreater(result["advertising_score"], 0.0)
        self.assertGreater(result["social_media_score"], 0.0)

    def test_unrelated_clean_content_scores_zero(self) -> None:
        result = analyze_promotion(
            "Brihadeeswarar temple inscription",
            "Ancient Chola heritage",
            {"temple": 0.95},
            [{"class": "person", "confidence": 0.8}],
        )

        self.assertEqual(result["promotion_score"], 0.0)
        self.assertEqual(result["advertising_score"], 0.0)
        self.assertEqual(result["affiliate_score"], 0.0)
        self.assertEqual(result["social_media_score"], 0.0)

    def test_compatibility_alias_matches_primary_api(self) -> None:
        args = (
            "Sponsored",
            "follow us",
            {"brand promotion": 0.7},
            [{"class": "book", "confidence": 0.5}],
        )

        self.assertEqual(get_promotion_scores(*args), analyze_promotion(*args))
