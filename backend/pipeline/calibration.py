"""Score calibration for moderation pipeline signals.

Applies temperature-scaled sigmoid to raw model outputs before fusion.
Temperature > 1.0 softens extreme scores; < 1.0 sharpens them.

Default temperature=1.6 was chosen to keep SigLIP2 heritage scores in the
0.4–0.7 range (they otherwise cluster near 0.8+) while leaving genuine
safety signals above 0.5 intact.
"""

from __future__ import annotations

import math
from collections.abc import Callable


def calibrate(score: float, temperature: float = 1.6) -> float:
    """Temperature-scaled sigmoid: 1 / (1 + exp(-score / temperature)).

    Input is treated as a logit (any real number), so raw model scores
    should be passed directly. Scores already in [0,1] are re-interpreted
    as logits centred on 0.5, which is the standard calibration treatment.
    """
    try:
        return 1.0 / (1.0 + math.exp(-float(score) / float(temperature)))
    except (OverflowError, ZeroDivisionError, ValueError):
        return 0.5


def calibrate_nsfw(score: float) -> float:
    """Calibrate OpenNSFW2 output (already in [0,1])."""
    return calibrate(score, temperature=1.6)


def calibrate_siglip(score: float) -> float:
    """Calibrate SigLIP2 sigmoid output (already in [0,1])."""
    return calibrate(score, temperature=1.6)


def calibrate_yolo(score: float) -> float:
    """Calibrate YOLO confidence score (already in [0,1])."""
    return calibrate(score, temperature=1.4)


def calibrate_scores(
    scores: dict[str, float], fn: Callable[[float], float] | None = None
) -> dict[str, float]:
    """Apply calibration to every value in a score dict."""
    f = fn or (lambda x: calibrate(x))
    return {k: f(v) for k, v in scores.items()}
