"""AMD XDNA NPU runtime backend through ONNX Runtime."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from core.runtime.detector import RuntimeInfo

logger = logging.getLogger(__name__)

VITIS_AI_PROVIDER = "VitisAIExecutionProvider"


def _provider_options() -> dict[str, Any]:
    """Build optional VitisAI provider options without requiring NPU config."""
    options: dict[str, Any] = {}
    config_path = os.getenv("VITISAI_CONFIG") or os.getenv("RYZEN_AI_CONFIG")
    if config_path and Path(config_path).exists():
        options["config_file"] = config_path

    raw_options = os.getenv("VITISAI_PROVIDER_OPTIONS")
    if raw_options:
        try:
            parsed = json.loads(raw_options)
            if isinstance(parsed, dict):
                options.update(parsed)
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid VITISAI_PROVIDER_OPTIONS JSON")
    return options


def detect() -> RuntimeInfo | None:
    """Return AMD XDNA NPU runtime information when ONNX Runtime exposes VitisAI."""
    try:
        import onnxruntime as ort
    except Exception:
        logger.debug("onnxruntime is unavailable; AMD XDNA NPU runtime cannot be selected")
        return None

    try:
        providers = tuple(ort.get_available_providers())
    except Exception:
        logger.debug("Unable to inspect ONNX Runtime providers", exc_info=True)
        return None

    if VITIS_AI_PROVIDER not in providers:
        return None

    execution_providers = (VITIS_AI_PROVIDER, "CPUExecutionProvider")
    return RuntimeInfo(
        name="npu",
        device="npu",
        provider=VITIS_AI_PROVIDER,
        supports_torch=False,
        supports_onnx=True,
        execution_providers=execution_providers,
        provider_options=_provider_options(),
        metadata={"onnxruntime_providers": providers},
    )
