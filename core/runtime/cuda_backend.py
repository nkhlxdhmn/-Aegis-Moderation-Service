"""NVIDIA CUDA runtime backend."""

from __future__ import annotations

import logging
import os
from typing import Any

from core.runtime.detector import RuntimeInfo

logger = logging.getLogger(__name__)


def _requested_cuda_device(torch: Any) -> str:
    requested = os.getenv("AEGIS_CUDA_DEVICE") or os.getenv("VLM_DEVICE") or "cuda:0"
    if not str(requested).startswith("cuda"):
        return "cuda:0"
    device_count = int(torch.cuda.device_count()) if hasattr(torch.cuda, "device_count") else 1
    try:
        index = int(str(requested).split(":", 1)[1]) if ":" in str(requested) else 0
    except ValueError:
        index = 0
    if index >= device_count:
        logger.warning(
            "Requested CUDA device %s but only %d device(s) are visible; using cuda:0",
            requested,
            device_count,
        )
        return "cuda:0"
    return f"cuda:{index}"


def _onnx_cuda_providers() -> tuple[str, ...]:
    try:
        import onnxruntime as ort

        providers = tuple(ort.get_available_providers())
    except Exception:
        return ()
    return tuple(provider for provider in providers if provider == "CUDAExecutionProvider")


def detect() -> RuntimeInfo | None:
    """Return CUDA runtime information when NVIDIA CUDA is available."""
    try:
        import torch
    except Exception:
        logger.debug("PyTorch is unavailable; CUDA runtime cannot be selected", exc_info=True)
        return None

    try:
        if not torch.cuda.is_available():
            return None
        device = _requested_cuda_device(torch)
        providers = _onnx_cuda_providers() or ("CUDAExecutionProvider",)
        device_index = int(device.split(":", 1)[1]) if ":" in device else 0
        if hasattr(torch.cuda, "get_device_name"):
            device_name = str(torch.cuda.get_device_name(device_index))
        else:
            device_name = "NVIDIA CUDA"
        device_count = int(torch.cuda.device_count()) if hasattr(torch.cuda, "device_count") else 1
        return RuntimeInfo(
            name="cuda",
            device=device,
            provider="CUDAExecutionProvider",
            supports_torch=True,
            supports_onnx=True,
            execution_providers=providers,
            metadata={"device_name": device_name, "device_count": device_count},
        )
    except Exception:
        logger.debug("CUDA runtime detection failed", exc_info=True)
        return None
