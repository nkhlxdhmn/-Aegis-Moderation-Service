"""ML toxicity — disabled until a PyTorch ≥ 2.6 image is in use.

All public multilingual toxicity models on HuggingFace ship weights in
PyTorch `.bin` (pickle) format, which transformers now blocks on PyTorch < 2.6
(CVE-2025-32434).  The Dockerfile pins torch==2.5.1+cu124 so no compatible
model can be loaded without a full image rebuild.

Hate speech and toxicity are fully covered by:
  - hard_block.py   — zero-tolerance keywords (runs before GPU inference)
  - text_safety.py  — rule-based hate/harassment/political detection
  - decision_engine — Tier 0-D thresholds on those rule scores

When torch is upgraded to ≥ 2.6 in the Dockerfile, re-enable this module by
replacing `_DISABLED = True` with `_DISABLED = False` and setting _MODEL_ID.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DISABLED = True
_pipeline = None  # re-enable by setting _DISABLED = False and assigning a loaded HF pipeline


def _get_pipeline():
    return None


def analyze(ocr_text: str | None, caption: str | None) -> dict[str, float]:
    return {"ml_toxicity_score": 0.0, "ml_hate_score": 0.0}
