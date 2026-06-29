"""CPU runtime backend."""

from __future__ import annotations

from core.runtime.detector import RuntimeInfo


def detect() -> RuntimeInfo:
    """Return the always-available CPU fallback runtime."""
    return RuntimeInfo(
        name="cpu",
        device="cpu",
        provider="CPUExecutionProvider",
        supports_torch=True,
        supports_onnx=True,
        execution_providers=("CPUExecutionProvider",),
    )
