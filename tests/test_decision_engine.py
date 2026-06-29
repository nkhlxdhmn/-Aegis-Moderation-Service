"""Decision-engine tests for moderation thresholds and priority."""

from unittest import TestCase

from backend.pipeline import decision_engine as engine


def _scores(**overrides: float) -> dict[str, float]:
    scores = {
        "adult_score": 0.0,
        "heritage_score": 0.0,
        "content_quality_score": 0.0,
        "child_safety_score": 0.0,
        "child_presence_score": 0.0,
        "terrorism_score": 0.0,
        "drug_trafficking_score": 0.0,
        "human_trafficking_score": 0.0,
        "violence_self_harm_score": 0.0,
        "weapon_score": 0.0,
        "blood_score": 0.0,
        "self_harm_score": 0.0,
        "promotion_score": 0.0,
        "advertising_score": 0.0,
        "affiliate_score": 0.0,
        "social_media_score": 0.0,
        "fraud_score": 0.0,
        "hate_speech_score": 0.0,
        "harassment_score": 0.0,
        "misinformation_score": 0.0,
        "self_harm_text_score": 0.0,
        "pii_score": 0.0,
    }
    scores.update(overrides)
    return scores


class DecisionEngineTests(TestCase):
    def assertDecision(self, scores: dict[str, float], decision: str, code: str) -> None:
        actual_decision, reason = engine.decide(scores)

        self.assertEqual(actual_decision, decision)
        self.assertEqual(engine.get_reason_code(scores), code)
        self.assertTrue(reason)

    def test_thresholds_are_strictly_greater_than_configured_values(self) -> None:
        # At exactly the threshold value the tier-specific check (which uses >)
        # does NOT fire, so the content is APPROVED.
        cases = (
            ("child_safety_score", engine.CHILD_SAFETY_THRESHOLD),
            ("terrorism_score", engine.TERRORISM_THRESHOLD),
            ("drug_trafficking_score", engine.DRUG_TRAFFICKING_THRESHOLD),
            ("human_trafficking_score", engine.HUMAN_TRAFFICKING_THRESHOLD),
            ("violence_self_harm_score", engine.VIOLENCE_SELF_HARM_THRESHOLD),
            ("weapon_score", engine.WEAPON_THRESHOLD),
            ("blood_score", engine.BLOOD_THRESHOLD),
            ("self_harm_score", engine.SELF_HARM_THRESHOLD),
            ("self_harm_text_score", engine.SELF_HARM_THRESHOLD),
            ("adult_score", engine.ADULT_REJECT_THRESHOLD),
            ("pii_score", engine.PRIVACY_THRESHOLD),
            ("fraud_score", engine.FRAUD_THRESHOLD),
            ("promotion_score", engine.PROMOTION_THRESHOLD),
            ("content_quality_score", engine.CONTENT_QUALITY_THRESHOLD),
        )

        for key, threshold in cases:
            with self.subTest(key=key):
                self.assertDecision(_scores(**{key: threshold}), "APPROVED", engine.APPROVED)

        # hate_speech at exactly threshold: Tier-0D uses >=, so REJECTED (not APPROVED).
        with self.subTest(key="hate_speech_score"):
            decision, _ = engine.decide(_scores(hate_speech_score=engine.HATE_SPEECH_THRESHOLD))
            self.assertEqual(decision, "REJECTED")

        # harassment at exactly threshold: Tier-6 uses >, doesn't fire.
        # Tier-10 confidence gate catches the elevated signal → UNDER_REVIEW.
        with self.subTest(key="harassment_score"):
            decision, _ = engine.decide(_scores(harassment_score=engine.HARASSMENT_THRESHOLD))
            self.assertEqual(decision, "UNDER_REVIEW")

    def test_child_safety_has_top_priority(self) -> None:
        # Child safety (Tier-1) fires before terrorism (Tier-3).
        # adult_score must stay below EXPLICIT_NSFW_THRESHOLD to avoid
        # Tier-0A firing before the child safety check.
        self.assertDecision(
            _scores(
                child_safety_score=engine.CHILD_SAFETY_REVIEW_THRESHOLD + 0.01,
                terrorism_score=1.0,
            ),
            "UNDER_REVIEW",
            engine.CHILD_SAFETY_ALERT,
        )

    def test_terrorism_and_illegal_content_reason_codes(self) -> None:
        cases = (
            (
                {"drug_trafficking_score": engine.DRUG_TRAFFICKING_THRESHOLD + 0.01},
                engine.DRUG_TRAFFICKING_CONTENT,
            ),
            (
                {"human_trafficking_score": engine.HUMAN_TRAFFICKING_THRESHOLD + 0.01},
                engine.HUMAN_TRAFFICKING_CONTENT,
            ),
            (
                {"terrorism_score": engine.TERRORISM_THRESHOLD + 0.01},
                engine.TERRORISM_CONTENT,
            ),
        )

        for overrides, code in cases:
            with self.subTest(code=code):
                self.assertDecision(_scores(**overrides), "REJECTED", code)

    def test_violence_self_harm_and_weapon_reason_codes(self) -> None:
        cases = (
            (
                {"self_harm_score": engine.SELF_HARM_THRESHOLD + 0.01},
                engine.SELF_HARM_CONTENT,
            ),
            (
                {"self_harm_text_score": engine.SELF_HARM_THRESHOLD + 0.01},
                engine.SELF_HARM_CONTENT,
            ),
            (
                {"violence_self_harm_score": engine.VIOLENCE_SELF_HARM_THRESHOLD + 0.01},
                engine.VIOLENCE_CONTENT,
            ),
            (
                {"blood_score": engine.BLOOD_THRESHOLD + 0.01},
                engine.VIOLENCE_CONTENT,
            ),
            (
                {"weapon_score": engine.WEAPON_THRESHOLD + 0.01},
                engine.WEAPON_CONTENT,
            ),
        )

        for overrides, code in cases:
            with self.subTest(code=code):
                self.assertDecision(_scores(**overrides), "REJECTED", code)

    def test_adult_content_and_heritage_exception(self) -> None:
        self.assertDecision(
            _scores(adult_score=engine.ADULT_REJECT_THRESHOLD + 0.01),
            "REJECTED",
            engine.NSFW_CONTENT,
        )
        self.assertDecision(
            _scores(
                adult_score=engine.ADULT_REJECT_THRESHOLD + 0.01,
                heritage_score=engine.HERITAGE_REVIEW_THRESHOLD + 0.01,
            ),
            "UNDER_REVIEW",
            engine.HERITAGE_REVIEW,
        )

    def test_privacy_fraud_hate_harassment_and_promotion_reason_codes(self) -> None:
        cases = (
            (
                {"pii_score": engine.PRIVACY_THRESHOLD + 0.01},
                "UNDER_REVIEW",
                engine.PRIVACY_VIOLATION,
            ),
            (
                {"fraud_score": engine.FRAUD_THRESHOLD + 0.01},
                "REJECTED",
                engine.FRAUD_CONTENT,
            ),
            (
                # hate_speech_score >= HATE_SPEECH_RULE_THRESHOLD triggers Tier-0D
                # (before Tier-6), so the reason code is HATE_SPEECH_REJECTION.
                {"hate_speech_score": engine.HATE_SPEECH_THRESHOLD + 0.01},
                "REJECTED",
                engine.HATE_SPEECH_REJECTION,
            ),
            (
                {"harassment_score": engine.HARASSMENT_THRESHOLD + 0.01},
                "REJECTED",
                engine.HARASSMENT,
            ),
            (
                {"promotion_score": engine.PROMOTION_THRESHOLD + 0.01},
                "UNDER_REVIEW",
                engine.PROMOTION_CONTENT,
            ),
        )

        for overrides, decision, code in cases:
            with self.subTest(code=code):
                self.assertDecision(_scores(**overrides), decision, code)

    def test_content_quality_and_approved_reason_codes(self) -> None:
        self.assertDecision(
            _scores(content_quality_score=engine.CONTENT_QUALITY_THRESHOLD + 0.01),
            "UNDER_REVIEW",
            engine.LOW_CONTENT_QUALITY,
        )
        self.assertDecision(_scores(), "APPROVED", engine.APPROVED)

    def test_priority_order_is_enforced_after_child_safety(self) -> None:
        # Each pair uses values that stay within the "soft" tier range to avoid
        # triggering Tier-0 hard rejects before the tier under test.
        # violence_self_harm stays below VIOLENCE_HARD_THRESHOLD (0.65).
        # adult stays below EXPLICIT_NSFW_THRESHOLD (0.60).
        # promotion stays below PROMOTION_HARD_THRESHOLD (0.30).
        cases = (
            # terrorism (Tier-3) before violence (Tier-4)
            (
                _scores(
                    terrorism_score=engine.TERRORISM_THRESHOLD + 0.01,
                    violence_self_harm_score=engine.VIOLENCE_SELF_HARM_THRESHOLD + 0.01,
                ),
                engine.TERRORISM_CONTENT,
            ),
            # violence (Tier-4) before pii (Tier-6)
            # Note: combining violence + adult triggers Tier-0A secondary reject,
            # so pii is used as the lower-priority signal instead.
            (
                _scores(
                    violence_self_harm_score=engine.VIOLENCE_SELF_HARM_THRESHOLD + 0.01,
                    pii_score=engine.PRIVACY_THRESHOLD + 0.01,
                ),
                engine.VIOLENCE_CONTENT,
            ),
            # adult (Tier-5) before pii (Tier-6)
            (
                _scores(
                    adult_score=engine.ADULT_REJECT_THRESHOLD + 0.01,
                    pii_score=engine.PRIVACY_THRESHOLD + 0.01,
                ),
                engine.NSFW_CONTENT,
            ),
            # pii before fraud (both Tier-6, pii checked first)
            (
                _scores(
                    pii_score=engine.PRIVACY_THRESHOLD + 0.01,
                    fraud_score=engine.FRAUD_THRESHOLD + 0.01,
                ),
                engine.PRIVACY_VIOLATION,
            ),
            # Tier-0D hate speech fires before Tier-6 fraud
            (
                _scores(
                    hate_speech_score=engine.HATE_SPEECH_RULE_THRESHOLD,
                    fraud_score=engine.FRAUD_THRESHOLD + 0.01,
                ),
                engine.HATE_SPEECH_REJECTION,
            ),
            # harassment (Tier-6) before promotion (Tier-7)
            (
                _scores(
                    harassment_score=engine.HARASSMENT_THRESHOLD + 0.01,
                    promotion_score=engine.PROMOTION_THRESHOLD + 0.01,
                ),
                engine.HARASSMENT,
            ),
            # promotion before content quality (both Tier-7, promotion checked first)
            (
                _scores(
                    promotion_score=engine.PROMOTION_THRESHOLD + 0.01,
                    content_quality_score=engine.CONTENT_QUALITY_THRESHOLD + 0.01,
                ),
                engine.PROMOTION_CONTENT,
            ),
            # content quality (Tier-7) fires before heritage exception (checked later)
            (
                _scores(
                    content_quality_score=engine.CONTENT_QUALITY_THRESHOLD + 0.01,
                    adult_score=engine.ADULT_REJECT_THRESHOLD + 0.01,
                    heritage_score=engine.HERITAGE_REVIEW_THRESHOLD + 0.01,
                ),
                engine.LOW_CONTENT_QUALITY,
            ),
        )

        for scores, code in cases:
            with self.subTest(code=code):
                self.assertEqual(engine.get_reason_code(scores), code)
