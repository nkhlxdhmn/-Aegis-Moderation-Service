"""Decision engine for the Aegis moderation pipeline.

Pure rule-based design â€” no LLM dependency:
  - Higher thresholds to reduce false positives for cultural/heritage content.
  - Cultural protection: heritage content gets effective score reductions
    inside the tier checks (raw scores preserved for audit trail).
  - Disagreement detection: when risk and heritage signals both run high
    (ambiguous cultural context), route to human review instead of auto-reject.

Evaluation stops at the first matching rule so severe safety risks override
later quality checks.

Tier 0 hard-rejects:
  - Explicit pornography, extreme violence, and promotional spam are rejected
    immediately, bypassing heritage exceptions.
  - These rules fire first to guarantee no approval path exists for the
    most severe categories.
"""

import logging

# â”€â”€ Per-category thresholds (Phase 3 â€” false-positive reduction) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ADULT_REJECT_THRESHOLD = 0.65
HERITAGE_REVIEW_THRESHOLD = 0.50
CONTENT_QUALITY_THRESHOLD = 0.35
CHILD_SAFETY_THRESHOLD = 0.45
CHILD_SAFETY_REVIEW_THRESHOLD = 0.65
TERRORISM_THRESHOLD = 0.60
DRUG_TRAFFICKING_THRESHOLD = 0.60
HUMAN_TRAFFICKING_THRESHOLD = 0.60
VIOLENCE_SELF_HARM_THRESHOLD = 0.65
WEAPON_THRESHOLD = 0.70
BLOOD_THRESHOLD = 0.65
SELF_HARM_THRESHOLD = 0.65
PRIVACY_THRESHOLD = 0.60
FRAUD_THRESHOLD = 0.60
HATE_SPEECH_THRESHOLD = 0.70
HARASSMENT_THRESHOLD = 0.65
PROMOTION_THRESHOLD = 0.70
LLAMA_REJECTION_CONFIDENCE = 0.92
LLAMA_APPROVAL_CONFIDENCE = 0.95

# â”€â”€ Tier 0 thresholds â€” hard rejects â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# NSFW / pornographic content
EXPLICIT_NSFW_THRESHOLD = 0.90
SECONDARY_ADULT_THRESHOLD = 0.45  # lowered
SECONDARY_COMPANION_THRESHOLD = 0.30  # lowered
VISUAL_EXPLICIT_BOOST = 0.10

# Violence / dangerous content
VIOLENCE_HARD_THRESHOLD = 0.90
GORE_HARD_THRESHOLD = 0.90
WEAPON_HARD_THRESHOLD = 0.90
WEAPON_VIOLENCE_WPN_THRESHOLD = 0.40  # lowered
WEAPON_VIOLENCE_VIO_THRESHOLD = 0.35  # lowered
WEAPON_GORE_WPN_THRESHOLD = 0.35  # lowered
WEAPON_GORE_BLD_THRESHOLD = 0.35  # lowered
VIOLENT_GORE_VIO_THRESHOLD = 0.35  # lowered
VIOLENT_GORE_BLD_THRESHOLD = 0.35  # lowered
SELF_HARM_HARD_THRESHOLD = 0.90

# Promotion / spam hard reject
PROMOTION_HARD_THRESHOLD = 0.90
SOCIAL_MEDIA_HARD_THRESHOLD = 0.30  # 1 social phrase (follow me, f4fâ€¦) â†’ REJECT
MARKETING_KEYWORD_MIN_COUNT = 1.0  # any single marketing keyword â†’ REJECT
COURSE_PROMOTION_THRESHOLD = 0.10  # any course promotion phrase â†’ REJECT
PHONE_NUMBER_REJECT_THRESHOLD = 0.35  # phone number in image â†’ REJECT
SOCIAL_HANDLE_REJECT_THRESHOLD = 0.25  # @handle in image â†’ REJECT
URL_REJECT_THRESHOLD = 0.35  # URL in image â†’ REJECT
QR_CODE_REJECT_THRESHOLD = 0.50  # QR code detected â†’ REJECT

# Tier 0-D: Hate speech / political rejection thresholds
ML_HATE_REJECT_THRESHOLD = 0.70  # ML hate score â†’ immediate REJECT
ML_TOXICITY_REJECT_THRESHOLD = 0.75  # ML toxicity â†’ immediate REJECT
ML_HATE_REVIEW_THRESHOLD = 0.50  # ML hate score â†’ UNDER_REVIEW
HATE_SPEECH_RULE_THRESHOLD = 0.55  # rule-based hate â†’ REJECT
POLITICAL_SCORE_THRESHOLD = 0.55  # lowered: catches more political content
POLITICAL_CAMPAIGN_THRESHOLD = 0.45  # lowered

# Video temporal safety
VIDEO_UNSAFE_RATIO_THRESHOLD = 0.15  # lowered
VIDEO_CONSECUTIVE_THRESHOLD = 3.0  # lowered

# Heritage exception: content with heritage_score â‰¥ this triggers UNDER_REVIEW
# instead of auto-REJECT for violence/weapon/self-harm/adult categories.
HERITAGE_EXCEPTION_THRESHOLD = 0.45

# Ensemble risk: above this â†’ UNDER_REVIEW
ENSEMBLE_RISK_THRESHOLD = 0.75

# Heritage score above which weapon/violence/child checks are softened
CULTURAL_PROTECTION_THRESHOLD = 0.65  # raised: require stronger heritage signal

# Heritage score + elevated risk (but not overwhelming) â†’ disagree â†’ review
_DISAGREE_RISK_LOW = 0.50  # risk must be > this to flag disagreement
_DISAGREE_RISK_HIGH = 0.90  # risk must be < this (overwhelming â†’ suppress disagree)
_DISAGREE_HERITAGE = 0.50  # heritage must be > this to flag disagreement

# â”€â”€ Reason codes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHILD_SAFETY_ALERT = "CHILD_SAFETY_ALERT"
TERRORISM_CONTENT = "TERRORISM_CONTENT"
DRUG_TRAFFICKING_CONTENT = "DRUG_TRAFFICKING_CONTENT"
HUMAN_TRAFFICKING_CONTENT = "HUMAN_TRAFFICKING_CONTENT"
VIOLENCE_CONTENT = "VIOLENCE_CONTENT"
SELF_HARM_CONTENT = "SELF_HARM_CONTENT"
WEAPON_CONTENT = "WEAPON_CONTENT"
NSFW_CONTENT = "NSFW_CONTENT"
PRIVACY_VIOLATION = "PRIVACY_VIOLATION"
FRAUD_CONTENT = "FRAUD_CONTENT"
HATE_SPEECH = "HATE_SPEECH"
HARASSMENT = "HARASSMENT"
PROMOTION_CONTENT = "PROMOTION_CONTENT"
LOW_CONTENT_QUALITY = "LOW_CONTENT_QUALITY"
HERITAGE_REVIEW = "HERITAGE_REVIEW"
PIPELINE_ERROR = "PIPELINE_ERROR"
ENSEMBLE_REVIEW = "ENSEMBLE_REVIEW"
DISAGREE_CONTENT = "DISAGREE_CONTENT"
HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
LLM_REJECTION = "LLM_REJECTION"
APPROVED = "APPROVED"

# â”€â”€ Tier 0 reason codes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EXPLICIT_PORNOGRAPHIC_CONTENT = "EXPLICIT_PORNOGRAPHIC_CONTENT"
HIGH_RISK_EXPLICIT_CONTENT = "HIGH_RISK_EXPLICIT_CONTENT"
HIGH_CONFIDENCE_VIOLENCE = "HIGH_CONFIDENCE_VIOLENCE"
GORE_CONTENT = "GORE_CONTENT"
DANGEROUS_WEAPON_CONTENT = "DANGEROUS_WEAPON_CONTENT"
WEAPON_VIOLENCE = "WEAPON_VIOLENCE"
WEAPON_GORE = "WEAPON_GORE"
VIOLENT_GORE = "VIOLENT_GORE"
ANIMAL_CRUELTY = "ANIMAL_CRUELTY"
HUMAN_KILLING = "HUMAN_KILLING"
PROMOTIONAL_SPAM = "PROMOTIONAL_SPAM"
MARKETING_CONTENT = "MARKETING_CONTENT"
COURSE_PROMOTION = "COURSE_PROMOTION"
POLITICAL_CAMPAIGN = "POLITICAL_CAMPAIGN"
HATE_SPEECH_REJECTION = "HATE_SPEECH_REJECTION"
TOXIC_CONTENT = "TOXIC_CONTENT"
QR_CODE_PROMOTION = "QR_CODE_PROMOTION"
PHONE_NUMBER_SPAM = "PHONE_NUMBER_SPAM"
SOCIAL_HANDLE_SPAM = "SOCIAL_HANDLE_SPAM"
URL_IN_IMAGE = "URL_IN_IMAGE"
VIDEO_UNSAFE_FRAMES = "VIDEO_UNSAFE_FRAMES"

# Uncertainty threshold â€” above this score routes to human review
UNCERTAINTY_REVIEW_THRESHOLD = 0.35

logger = logging.getLogger(__name__)


def _score(scores: dict[str, float], key: str) -> float:
    try:
        return float(scores.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _has_model_disagreement(
    child_safety_score: float,
    violence_score: float,
    weapon_score: float,
    heritage_score: float,
) -> bool:
    """Return True when non-adult risk signals and heritage disagree.

    Adult vs heritage disagreement is handled separately by the heritage
    exception in Tier 5.  This function catches weapon/violence/child cases
    where risk is elevated but not overwhelming, coexisting with high heritage.
    Sub-threshold but still elevated risk + high heritage = genuinely ambiguous.
    """
    max_non_adult_risk = max(child_safety_score, violence_score, weapon_score)
    return (
        max_non_adult_risk > _DISAGREE_RISK_LOW
        and max_non_adult_risk < _DISAGREE_RISK_HIGH
        and heritage_score > _DISAGREE_HERITAGE
    )


def _evaluate(scores: dict[str, float]) -> tuple[str, str, str]:
    """Return (decision, reason_code, human_reason)."""

    child_safety_score = _score(scores, "child_safety_score")
    terrorism_score = _score(scores, "terrorism_score")
    drug_trafficking_score = _score(scores, "drug_trafficking_score")
    human_trafficking_score = _score(scores, "human_trafficking_score")
    violence_self_harm_score = _score(scores, "violence_self_harm_score")
    weapon_score = _score(scores, "weapon_score")
    blood_score = _score(scores, "blood_score")
    self_harm_score = max(
        _score(scores, "self_harm_score"),
        _score(scores, "self_harm_text_score"),
    )
    adult_score = _score(scores, "adult_score")
    heritage_score = _score(scores, "heritage_score")
    pii_score = _score(scores, "pii_score")
    fraud_score = _score(scores, "fraud_score")
    hate_speech_score = _score(scores, "hate_speech_score")
    harassment_score = _score(scores, "harassment_score")
    promotion_score = _score(scores, "promotion_score")
    content_quality_score = _score(scores, "content_quality_score")
    ensemble_risk_score = _score(scores, "ensemble_risk_score")
    uncertainty_score = _score(scores, "uncertainty_score")

    # â”€â”€ Tier 0 supplemental inputs (populated by upstream pipeline) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    nsfw_score = _score(scores, "nsfw_score")
    visual_explicit = _score(scores, "visual_explicit_indicator")
    animal_cruelty_text = _score(scores, "animal_cruelty_text_score")
    human_killing_text = _score(scores, "human_killing_text_score")
    social_media_score_ = _score(scores, "social_media_score")
    marketing_keyword_count = _score(scores, "marketing_keyword_count")
    course_promotion_score_ = _score(scores, "course_promotion_score")
    phone_number_score_ = _score(scores, "phone_number_score")
    social_handle_score_ = _score(scores, "social_handle_score")
    url_score_ = _score(scores, "url_score")
    qr_code_score_ = _score(scores, "qr_code_score")
    ml_toxicity_score_ = _score(scores, "ml_toxicity_score")
    ml_hate_score_ = _score(scores, "ml_hate_score")
    political_score = _score(scores, "political_score")
    political_campaign_score = _score(scores, "political_campaign_score")
    video_unsafe_ratio = _score(scores, "video_unsafe_frame_ratio")
    video_consecutive = _score(scores, "video_consecutive_unsafe_frames")

    # â”€â”€ Tier 0-A: Hard NSFW / Pornographic reject â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Fires before Llama approval â€” explicit pornography has no approval path
    # and no heritage exception.
    adult_effective = max(adult_score, nsfw_score)
    if visual_explicit >= 0.5:
        adult_effective = min(1.0, adult_effective + VISUAL_EXPLICIT_BOOST)

    if adult_effective >= EXPLICIT_NSFW_THRESHOLD:
        logger.warning(
            "Tier-0A NSFW hard reject | adult_effective=%.3f adult=%.3f nsfw=%.3f decision=REJECTED reason=%s",
            adult_effective,
            adult_score,
            nsfw_score,
            EXPLICIT_PORNOGRAPHIC_CONTENT,
        )
        return (
            "REJECTED",
            EXPLICIT_PORNOGRAPHIC_CONTENT,
            "Rejected: explicit pornographic content detected with high confidence.",
        )

    if adult_score >= SECONDARY_ADULT_THRESHOLD and (
        child_safety_score >= SECONDARY_COMPANION_THRESHOLD
        or violence_self_harm_score >= SECONDARY_COMPANION_THRESHOLD
        or weapon_score >= SECONDARY_COMPANION_THRESHOLD
        or blood_score >= SECONDARY_COMPANION_THRESHOLD
    ):
        logger.warning(
            "Tier-0A secondary explicit reject | adult=%.3f nsfw=%.3f child=%.3f "
            "violence=%.3f weapon=%.3f blood=%.3f decision=REJECTED reason=%s",
            adult_score,
            nsfw_score,
            child_safety_score,
            violence_self_harm_score,
            weapon_score,
            blood_score,
            HIGH_RISK_EXPLICIT_CONTENT,
        )
        return (
            "REJECTED",
            HIGH_RISK_EXPLICIT_CONTENT,
            "Rejected: adult content combined with high-risk signal (violence, weapon, or child safety).",
        )

    # â”€â”€ Tier 0-B: Hard Violence / Dangerous content reject â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Fires before Llama approval â€” extreme violence bypasses heritage exception.
    violence_effective = max(violence_self_harm_score, blood_score, weapon_score, self_harm_score)

    if violence_self_harm_score >= VIOLENCE_HARD_THRESHOLD:
        logger.warning(
            "Tier-0B violence hard reject | effective=%.3f violence=%.3f blood=%.3f "
            "weapon=%.3f self_harm=%.3f decision=REJECTED reason=%s",
            violence_effective,
            violence_self_harm_score,
            blood_score,
            weapon_score,
            self_harm_score,
            HIGH_CONFIDENCE_VIOLENCE,
        )
        return (
            "REJECTED",
            VIOLENCE_CONTENT,
            "Rejected: high-confidence violent content detected.",
        )

    if blood_score >= GORE_HARD_THRESHOLD:
        logger.warning(
            "Tier-0B gore hard reject | blood=%.3f violence=%.3f weapon=%.3f "
            "self_harm=%.3f decision=REJECTED reason=%s",
            blood_score,
            violence_self_harm_score,
            weapon_score,
            self_harm_score,
            GORE_CONTENT,
        )
        return (
            "REJECTED",
            GORE_CONTENT,
            "Rejected: graphic gore or blood content detected.",
        )

    if weapon_score >= WEAPON_HARD_THRESHOLD:
        logger.warning(
            "Tier-0B weapon hard reject | weapon=%.3f violence=%.3f blood=%.3f "
            "self_harm=%.3f decision=REJECTED reason=%s",
            weapon_score,
            violence_self_harm_score,
            blood_score,
            self_harm_score,
            DANGEROUS_WEAPON_CONTENT,
        )
        return (
            "REJECTED",
            WEAPON_CONTENT,
            "Rejected: dangerous weapon content detected with high confidence.",
        )

    # Combined-signal checks use lower thresholds and therefore respect the heritage
    # exception: mythological battles, archery, and ritual scenes routinely cross
    # 0.50â€“0.60 on these dimensions while being culturally valid content.  The
    # absolute-threshold blocks above (0.85 / 0.80 / 0.85) are hard stops regardless.
    if (
        weapon_score >= WEAPON_VIOLENCE_WPN_THRESHOLD
        and violence_self_harm_score >= WEAPON_VIOLENCE_VIO_THRESHOLD
        and heritage_score < HERITAGE_EXCEPTION_THRESHOLD
    ):
        logger.warning(
            "Tier-0B weapon+violence reject | weapon=%.3f violence=%.3f decision=REJECTED reason=%s",
            weapon_score,
            violence_self_harm_score,
            WEAPON_VIOLENCE,
        )
        return (
            "REJECTED",
            WEAPON_VIOLENCE,
            "Rejected: combined weapon and violence signals exceed safe thresholds.",
        )

    if (
        weapon_score >= WEAPON_GORE_WPN_THRESHOLD
        and blood_score >= WEAPON_GORE_BLD_THRESHOLD
        and heritage_score < HERITAGE_EXCEPTION_THRESHOLD
    ):
        logger.warning(
            "Tier-0B weapon+gore reject | weapon=%.3f blood=%.3f decision=REJECTED reason=%s",
            weapon_score,
            blood_score,
            WEAPON_GORE,
        )
        return (
            "REJECTED",
            WEAPON_GORE,
            "Rejected: combined weapon and gore signals exceed safe thresholds.",
        )

    if (
        violence_self_harm_score >= VIOLENT_GORE_VIO_THRESHOLD
        and blood_score >= VIOLENT_GORE_BLD_THRESHOLD
        and heritage_score < HERITAGE_EXCEPTION_THRESHOLD
    ):
        logger.warning(
            "Tier-0B violent gore reject | violence=%.3f blood=%.3f decision=REJECTED reason=%s",
            violence_self_harm_score,
            blood_score,
            VIOLENT_GORE,
        )
        return (
            "REJECTED",
            VIOLENT_GORE,
            "Rejected: combined violence and gore signals exceed safe thresholds.",
        )

    if animal_cruelty_text >= 0.5:
        logger.warning(
            "Tier-0B animal cruelty reject | animal_cruelty_text=%.3f decision=REJECTED reason=%s",
            animal_cruelty_text,
            ANIMAL_CRUELTY,
        )
        return (
            "REJECTED",
            ANIMAL_CRUELTY,
            "Rejected: animal cruelty or abuse content detected.",
        )

    if human_killing_text >= 0.5:
        logger.warning(
            "Tier-0B human killing reject | human_killing_text=%.3f decision=REJECTED reason=%s",
            human_killing_text,
            HUMAN_KILLING,
        )
        return (
            "REJECTED",
            HUMAN_KILLING,
            "Rejected: content depicting murder, execution, or lethal human violence.",
        )

    if self_harm_score >= SELF_HARM_HARD_THRESHOLD:
        logger.warning(
            "Tier-0B self-harm hard reject | self_harm=%.3f decision=REJECTED reason=%s",
            self_harm_score,
            SELF_HARM_CONTENT,
        )
        return (
            "REJECTED",
            SELF_HARM_CONTENT,
            "Rejected: self-harm content detected with high confidence.",
        )

    # â”€â”€ Tier 0-C: Promotion / Spam hard reject â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if promotion_score >= PROMOTION_HARD_THRESHOLD:
        logger.warning(
            "Tier-0C promotion spam reject | promotion=%.3f decision=REJECTED reason=%s",
            promotion_score,
            PROMOTIONAL_SPAM,
        )
        return (
            "REJECTED",
            PROMOTIONAL_SPAM,
            "Rejected: high-confidence promotional or spam content detected.",
        )

    if social_media_score_ >= SOCIAL_MEDIA_HARD_THRESHOLD:
        logger.warning(
            "Tier-0C social media spam reject | social=%.3f decision=REJECTED",
            social_media_score_,
        )
        return (
            "REJECTED",
            PROMOTIONAL_SPAM,
            "Rejected: social media follow/subscribe solicitation detected.",
        )

    if marketing_keyword_count >= MARKETING_KEYWORD_MIN_COUNT:
        logger.warning(
            "Tier-0C marketing keyword reject | keyword_count=%.0f decision=REJECTED reason=%s",
            marketing_keyword_count,
            MARKETING_CONTENT,
        )
        return (
            "REJECTED",
            MARKETING_CONTENT,
            "Rejected: multiple marketing or commercial keywords detected.",
        )

    if course_promotion_score_ >= COURSE_PROMOTION_THRESHOLD:
        logger.warning(
            "Tier-0C course promotion reject | course_score=%.3f decision=REJECTED reason=%s",
            course_promotion_score_,
            COURSE_PROMOTION,
        )
        return (
            "REJECTED",
            COURSE_PROMOTION,
            "Rejected: paid course or class marketing content detected.",
        )

    if (
        political_score >= POLITICAL_SCORE_THRESHOLD
        or political_campaign_score >= POLITICAL_CAMPAIGN_THRESHOLD
    ):
        logger.warning(
            "Tier-0C political reject | political=%.3f campaign=%.3f",
            political_score,
            political_campaign_score,
        )
        return (
            "REJECTED",
            POLITICAL_CAMPAIGN,
            "Rejected: political campaign, vote-solicitation, or party promotion detected.",
        )

    if phone_number_score_ >= PHONE_NUMBER_REJECT_THRESHOLD:
        logger.warning("Tier-0C phone number reject | score=%.3f", phone_number_score_)
        return (
            "REJECTED",
            PHONE_NUMBER_SPAM,
            "Rejected: phone number detected â€” promotional or spam content.",
        )

    if social_handle_score_ >= SOCIAL_HANDLE_REJECT_THRESHOLD:
        logger.warning("Tier-0C social handle reject | score=%.3f", social_handle_score_)
        return (
            "REJECTED",
            SOCIAL_HANDLE_SPAM,
            "Rejected: social media handle (@username) detected â€” promotional content.",
        )

    if url_score_ >= URL_REJECT_THRESHOLD:
        logger.warning("Tier-0C URL reject | score=%.3f", url_score_)
        return (
            "REJECTED",
            URL_IN_IMAGE,
            "Rejected: external URL detected in image.",
        )

    if qr_code_score_ >= QR_CODE_REJECT_THRESHOLD:
        logger.warning("Tier-0C QR code reject | score=%.3f", qr_code_score_)
        return (
            "REJECTED",
            QR_CODE_PROMOTION,
            "Rejected: QR code detected â€” likely links to external promotion or payment.",
        )

    # â”€â”€ Tier 0-D: Hate speech / ML toxicity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if ml_hate_score_ >= ML_HATE_REJECT_THRESHOLD:
        logger.warning(
            "Tier-0D ML hate reject | ml_hate=%.3f decision=REJECTED",
            ml_hate_score_,
        )
        return (
            "REJECTED",
            HATE_SPEECH_REJECTION,
            "Rejected: ML model detected hate speech or community targeting with high confidence.",
        )

    if ml_toxicity_score_ >= ML_TOXICITY_REJECT_THRESHOLD:
        logger.warning(
            "Tier-0D ML toxicity reject | ml_tox=%.3f decision=REJECTED",
            ml_toxicity_score_,
        )
        return (
            "REJECTED",
            TOXIC_CONTENT,
            "Rejected: ML model detected highly toxic content.",
        )

    if ml_hate_score_ >= ML_HATE_REVIEW_THRESHOLD:
        logger.warning(
            "Tier-0D ML hate review | ml_hate=%.3f decision=UNDER_REVIEW",
            ml_hate_score_,
        )
        return (
            "UNDER_REVIEW",
            HATE_SPEECH_REJECTION,
            "Flagged for review: elevated hate speech signal detected by ML model.",
        )

    rule_hate = _score(scores, "hate_speech_score")
    if rule_hate >= HATE_SPEECH_RULE_THRESHOLD:
        logger.warning(
            "Tier-0D rule hate reject | hate=%.3f decision=REJECTED",
            rule_hate,
        )
        return (
            "REJECTED",
            HATE_SPEECH_REJECTION,
            "Rejected: rule-based hate speech detection â€” religious, racial, or ethnic hatred.",
        )

    # â”€â”€ Tier 0-E: Video temporal safety rules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Fires before Llama approval â€” dense or sustained unsafe video content
    # cannot be approved regardless of individual frame scores.
    if video_unsafe_ratio >= VIDEO_UNSAFE_RATIO_THRESHOLD:
        logger.warning(
            "Tier-0D video unsafe ratio reject | ratio=%.3f decision=REJECTED reason=%s",
            video_unsafe_ratio,
            VIDEO_UNSAFE_FRAMES,
        )
        return (
            "REJECTED",
            VIDEO_UNSAFE_FRAMES,
            f"Rejected: {video_unsafe_ratio:.0%} of video frames contain unsafe content "
            f"(threshold {VIDEO_UNSAFE_RATIO_THRESHOLD:.0%}).",
        )

    if video_consecutive >= VIDEO_CONSECUTIVE_THRESHOLD:
        logger.warning(
            "Tier-0D video consecutive unsafe reject | consecutive=%.0f decision=REJECTED reason=%s",
            video_consecutive,
            VIDEO_UNSAFE_FRAMES,
        )
        return (
            "REJECTED",
            VIDEO_UNSAFE_FRAMES,
            f"Rejected: {video_consecutive:.0f} consecutive unsafe frames detected "
            f"(threshold {VIDEO_CONSECUTIVE_THRESHOLD:.0f}).",
        )

    is_heritage = heritage_score >= HERITAGE_EXCEPTION_THRESHOLD

    # â”€â”€ Cultural protection: heritage content gets effective score reductions â”€â”€
    # Raw scores are preserved in the result for audit; only the _eval variables
    # are used in threshold comparisons inside this function.
    # Phase 5: child_safety_eval always equals raw child_safety_score (child dominance).
    _h = heritage_score > CULTURAL_PROTECTION_THRESHOLD
    weapon_eval = weapon_score * 0.70 if _h else weapon_score
    blood_eval = blood_score * 0.70 if _h else blood_score
    violence_eval = violence_self_harm_score * 0.70 if _h else violence_self_harm_score
    self_harm_eval = self_harm_score * 0.70 if _h else self_harm_score
    child_safety_eval = child_safety_score  # Phase 5: heritage never reduces child safety

    # Dual child-safety threshold: borderline effective scores add a confidence
    # penalty to the ensemble risk; high scores route to review.
    child_penalty = (
        0.10 if CHILD_SAFETY_THRESHOLD <= child_safety_eval < CHILD_SAFETY_REVIEW_THRESHOLD else 0.0
    )
    effective_ensemble_risk = min(1.0, ensemble_risk_score + child_penalty)

    # â”€â”€ Tier 1: Child safety (dual threshold) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if child_safety_eval >= CHILD_SAFETY_REVIEW_THRESHOLD:
        return (
            "UNDER_REVIEW",
            CHILD_SAFETY_ALERT,
            "Child safety risk requires manual review.",
        )

    # â”€â”€ Tier 3: Trafficking / terrorism â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if drug_trafficking_score > DRUG_TRAFFICKING_THRESHOLD:
        return (
            "REJECTED",
            DRUG_TRAFFICKING_CONTENT,
            "Rejected because drug trafficking content exceeded the allowed threshold.",
        )
    if human_trafficking_score > HUMAN_TRAFFICKING_THRESHOLD:
        return (
            "REJECTED",
            HUMAN_TRAFFICKING_CONTENT,
            "Rejected because human trafficking content exceeded the allowed threshold.",
        )
    if terrorism_score > TERRORISM_THRESHOLD:
        return (
            "REJECTED",
            TERRORISM_CONTENT,
            "Rejected because terrorism or illegal-content risk exceeded the allowed threshold.",
        )

    # â”€â”€ Tier 4: Self-harm / violence / weapons (heritage-aware) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if self_harm_eval > SELF_HARM_THRESHOLD:
        if is_heritage:
            return (
                "UNDER_REVIEW",
                HERITAGE_REVIEW,
                "Heritage content with elevated self-harm signal requires manual review.",
            )
        return (
            "REJECTED",
            SELF_HARM_CONTENT,
            "Rejected because self-harm risk exceeded the allowed threshold.",
        )

    if violence_eval > VIOLENCE_SELF_HARM_THRESHOLD or blood_eval > BLOOD_THRESHOLD:
        if is_heritage:
            return (
                "UNDER_REVIEW",
                HERITAGE_REVIEW,
                "Heritage content with elevated violence signal requires manual review.",
            )
        return (
            "REJECTED",
            VIOLENCE_CONTENT,
            "Rejected because violence risk exceeded the allowed threshold.",
        )

    if weapon_eval > WEAPON_THRESHOLD:
        if is_heritage:
            return (
                "UNDER_REVIEW",
                HERITAGE_REVIEW,
                "Heritage content with elevated weapon signal requires manual review.",
            )
        return (
            "REJECTED",
            WEAPON_CONTENT,
            "Rejected because weapon-content risk exceeded the allowed threshold.",
        )

    # â”€â”€ Tier 5: Adult content (heritage exception) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    adult_content_detected = adult_score > ADULT_REJECT_THRESHOLD
    heritage_exception_candidate = (
        adult_content_detected and heritage_score > HERITAGE_REVIEW_THRESHOLD
    )
    if adult_content_detected and not heritage_exception_candidate:
        return (
            "REJECTED",
            NSFW_CONTENT,
            "Rejected because adult-content risk exceeded the allowed threshold without sufficient heritage relevance.",
        )

    # â”€â”€ Tier 6: Privacy / fraud / hate / harassment â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if pii_score > PRIVACY_THRESHOLD:
        return (
            "UNDER_REVIEW",
            PRIVACY_VIOLATION,
            "Privacy or personally identifiable information risk requires manual review.",
        )
    if fraud_score > FRAUD_THRESHOLD:
        return (
            "REJECTED",
            FRAUD_CONTENT,
            "Rejected because fraud or scam risk exceeded the allowed threshold.",
        )
    if hate_speech_score > HATE_SPEECH_THRESHOLD:
        return (
            "REJECTED",
            HATE_SPEECH,
            "Rejected because hate-speech risk exceeded the allowed threshold.",
        )
    if harassment_score > HARASSMENT_THRESHOLD:
        return (
            "REJECTED",
            HARASSMENT,
            "Rejected because harassment or targeted-abuse risk exceeded the allowed threshold.",
        )

    # â”€â”€ Tier 7: Quality / promotion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if promotion_score > PROMOTION_THRESHOLD:
        return (
            "UNDER_REVIEW",
            PROMOTION_CONTENT,
            "Promotion or advertising risk exceeded the automatic approval threshold.",
        )
    if content_quality_score > CONTENT_QUALITY_THRESHOLD:
        return (
            "UNDER_REVIEW",
            LOW_CONTENT_QUALITY,
            "Content quality risk exceeded the automatic approval threshold.",
        )

    # â”€â”€ Tier 8: Ensemble catch-all â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if effective_ensemble_risk > ENSEMBLE_RISK_THRESHOLD:
        return (
            "UNDER_REVIEW",
            ENSEMBLE_REVIEW,
            "Combined model risk score is elevated; routed for human review.",
        )

    # â”€â”€ Tier 8.5: High uncertainty â†’ route to human review â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if uncertainty_score > UNCERTAINTY_REVIEW_THRESHOLD:
        return (
            "UNDER_REVIEW",
            HIGH_UNCERTAINTY,
            "High model uncertainty detected; routed for human review.",
        )

    # â”€â”€ Tier 9: Disagreement detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Catches sub-threshold but elevated risk coexisting with high heritage,
    # e.g. mythological battle scenes, fire rituals, festival children.
    if _has_model_disagreement(child_safety_eval, violence_eval, weapon_eval, heritage_score):
        return (
            "UNDER_REVIEW",
            DISAGREE_CONTENT,
            "Elevated risk signal alongside high heritage score indicates ambiguous cultural content; routed for human review.",
        )

    # â”€â”€ Heritage review for borderline adult + heritage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if heritage_exception_candidate:
        return (
            "UNDER_REVIEW",
            HERITAGE_REVIEW,
            "Adult-content risk detected alongside strong heritage relevance; routed for heritage exception review.",
        )

    # â”€â”€ Tier 10: Confidence gate â€” never auto-approve uncertain content â”€â”€â”€â”€â”€â”€â”€â”€
    # If ANY elevated signal exists but stayed below hard-reject thresholds,
    # route to REVIEW rather than APPROVE.
    _any_elevated = max(
        ml_toxicity_score_,
        ml_hate_score_,
        _score(scores, "hate_speech_score"),
        _score(scores, "harassment_score"),
        political_score,
        political_campaign_score,
        phone_number_score_,
        social_handle_score_,
        url_score_,
        qr_code_score_,
    )
    if _any_elevated >= 0.20:
        logger.info(
            "Tier-10 confidence gate â†’ UNDER_REVIEW (any_elevated=%.3f)",
            _any_elevated,
        )
        return (
            "UNDER_REVIEW",
            ENSEMBLE_REVIEW,
            "One or more signals are elevated but below hard-reject; routed for human review.",
        )

    if uncertainty_score >= 0.25:
        logger.info(
            "Tier-10 uncertainty gate â†’ UNDER_REVIEW (uncertainty=%.3f)", uncertainty_score
        )
        return (
            "UNDER_REVIEW",
            HIGH_UNCERTAINTY,
            "Low model confidence â€” routed for human review rather than automatic approval.",
        )

    return (
        "APPROVED",
        APPROVED,
        "Approved: all safety, promotion, hate speech, and quality checks passed.",
    )


def decide(scores: dict[str, float]) -> tuple[str, str]:
    """Return (decision, human_reason)."""
    decision, _, reason = _evaluate(scores)
    return decision, reason


def get_reason_code(scores: dict[str, float]) -> str:
    _, reason_code, _ = _evaluate(scores)
    return reason_code


def decide_with_reason_code(scores: dict[str, float]) -> tuple[str, str, str]:
    """Return (decision, reason_code, human_reason)."""
    return _evaluate(scores)
