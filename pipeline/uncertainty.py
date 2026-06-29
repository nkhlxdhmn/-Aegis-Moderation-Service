"""
uncertainty.py – Multi-source uncertainty estimation for moderation decisions.

Combines four orthogonal signals to produce a single uncertainty score in
[0, 1]:

    uncertainty = 0.30 * score_variance
                + 0.30 * score_entropy
                + 0.20 * (1 - caption_similarity)
                + 0.20 * model_disagreement
"""

import logging
import math

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIGNAL_KEYS: tuple[str, ...] = (
    "adult_score",
    "child_safety_score",
    "violence_self_harm_score",
    "weapon_score",
    "fraud_score",
    "terrorism_score",
)

_WEIGHTS = {
    "score_variance": 0.30,
    "score_entropy": 0.30,
    "caption_dissimilarity": 0.20,
    "model_disagreement": 0.20,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_uncertainty(
    scores: dict[str, float],
    captions: list[str],
    llama_result: dict | None,
    qwen_description: str = "",
) -> float:
    """Compute a composite uncertainty score for a moderation decision.

    Parameters
    ----------
    scores:
        Dict of risk scores keyed by signal name.  Missing keys default to
        0.0.  The six canonical signals defined in ``_SIGNAL_KEYS`` are used.
    captions:
        List of caption strings produced by different vision models.  Used
        to estimate inter-model caption agreement.
    llama_result:
        Optional dict from the Llama-based moderation model with at least
        ``"decision"`` (``"APPROVED"``/``"REJECTED"``/``"UNDER_REVIEW"``)
        and ``"confidence"`` (float in [0, 1]).
    qwen_description:
        Free-form description from the Qwen-VL model (currently reserved for
        future use; not included in the formula).

    Returns
    -------
    float
        Uncertainty in [0, 1].  Higher → less certain.
    """
    signals = _extract_signals(scores)

    sv = _score_variance(signals)
    se = _score_entropy(signals)
    cs = _caption_similarity(captions)
    md = _model_disagreement(signals, llama_result)

    uncertainty = (
        _WEIGHTS["score_variance"] * sv
        + _WEIGHTS["score_entropy"] * se
        + _WEIGHTS["caption_dissimilarity"] * (1.0 - cs)
        + _WEIGHTS["model_disagreement"] * md
    )

    result = min(1.0, max(0.0, uncertainty))

    logger.debug(
        "uncertainty: score_variance=%.4f score_entropy=%.4f "
        "caption_similarity=%.4f model_disagreement=%.4f → %.4f",
        sv,
        se,
        cs,
        md,
        result,
    )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_signals(scores: dict[str, float]) -> list[float]:
    """Return the six canonical signal values (missing keys → 0.0)."""
    return [float(scores.get(key, 0.0)) for key in _SIGNAL_KEYS]


def _score_variance(signals: list[float]) -> float:
    """Normalised population variance of the six risk signals.

    The maximum possible variance for a binary 0/1 distribution is 0.25, so
    multiplying by 4 maps that range to [0, 1].
    """
    n = len(signals)
    if n == 0:
        return 0.0
    mean = sum(signals) / n
    variance = sum((x - mean) ** 2 for x in signals) / n
    return min(1.0, variance * 4.0)


def _score_entropy(signals: list[float]) -> float:
    """Normalised Shannon entropy of the six risk signals treated as a PMF.

    If all signals are zero the entropy is defined as 0.0 (no information).
    Normalisation is by log2(6) so the result lies in [0, 1].
    """
    total = sum(signals) + 1e-10
    if total <= 1e-10 or all(s == 0.0 for s in signals):
        return 0.0

    log2_n = math.log2(len(signals))
    if log2_n == 0:
        return 0.0

    entropy = 0.0
    for s in signals:
        p = s / total
        if p > 1e-10:
            entropy -= p * math.log2(p)

    return min(1.0, max(0.0, entropy / log2_n))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _caption_similarity(captions: list[str]) -> float:
    """Mean pairwise Jaccard similarity of caption word sets.

    Returns
    -------
    float
        1.0  if only one caption (self-agreement by definition)
        0.5  if no captions (unknown)
        mean pairwise Jaccard otherwise
    """
    if len(captions) == 0:
        return 0.5
    if len(captions) == 1:
        return 1.0

    token_sets = [set(c.lower().split()) for c in captions]

    scores: list[float] = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            scores.append(_jaccard(token_sets[i], token_sets[j]))

    return sum(scores) / len(scores) if scores else 1.0


def _model_disagreement(
    signals: list[float],
    llama_result: dict | None,
) -> float:
    """Measure disagreement between risk scores and the Llama-based decision.

    Returns a value in [0, 1].
    """
    result = llama_result or {}
    llama_decision: str = str(result.get("decision", "UNDER_REVIEW")).upper()
    llama_confidence: float = float(result.get("confidence", 0.5))
    # Clamp confidence to a valid range.
    llama_confidence = min(1.0, max(0.0, llama_confidence))

    max_risk: float = max(signals) if signals else 0.0

    if llama_decision == "APPROVED" and max_risk > 0.50:
        disagreement = max_risk
    elif llama_decision == "REJECTED" and max_risk < 0.30:
        disagreement = 1.0 - max_risk
    else:
        # Partial disagreement: scale by 0.5 to keep it subdominant.
        if llama_decision == "REJECTED":
            disagreement = abs(max_risk - llama_confidence) * 0.5
        else:
            disagreement = abs(max_risk - (1.0 - llama_confidence)) * 0.5

    return min(1.0, max(0.0, disagreement))
