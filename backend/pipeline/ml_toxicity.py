"""ML toxicity - optional Hugging Face toxicity model integration.

This module remains disabled by default because the rule-based text safety
stack already covers the active moderation categories, and enabling another
transformers pipeline would add model downloads and memory pressure.

Hate speech and toxicity are covered by:
  - hard_block.py - zero-tolerance keywords
  - text_safety.py - rule-based hate/harassment/political detection
  - decision_engine - thresholds on those rule scores

To enable it later, replace `_DISABLED = True` with `_DISABLED = False` and
set _MODEL_ID to the selected toxicity model.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DISABLED = True
_pipeline = None


def _get_pipeline():
    return None


def analyze(ocr_text: str | None, caption: str | None) -> dict[str, float]:
    return {"ml_toxicity_score": 0.0, "ml_hate_score": 0.0}
