"""Runtime backend selection for model inference.

Priority:
  1. NVIDIA CUDA
  2. AMD XDNA NPU through ONNX Runtime VitisAIExecutionProvider
  3. CPU fallback

The NPU backend is intentionally optional. If ONNX Runtime or the AMD VitisAI
execution provider is not installed, detection silently falls back to CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RuntimeName = Literal["cuda", "npu", "cpu"]


@dataclass(frozen=True)
class RuntimeInfo:
    """Selected hardware runtime and the capabilities exposed by it."""

    name: RuntimeName
    device: str
    provider: str
    supports_torch: bool
    supports_onnx: bool
    execution_providers: tuple[str, ...] = ()
    provider_options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def torch_device(self) -> str:
        """Return a torch-compatible device, falling back to CPU for ONNX-only runtimes."""
        return self.device if self.supports_torch else "cpu"

    @property
    def is_accelerated(self) -> bool:
        return self.name in {"cuda", "npu"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
            "torch_device": self.torch_device,
            "provider": self.provider,
            "supports_torch": self.supports_torch,
            "supports_onnx": self.supports_onnx,
            "execution_providers": list(self.execution_providers),
            "provider_options": dict(self.provider_options),
            "metadata": dict(self.metadata),
        }


def select_runtime() -> RuntimeInfo:
    """Detect and return the preferred available runtime."""
    from core.runtime import cpu_backend, cuda_backend, npu_backend

    for detector in (cuda_backend.detect, npu_backend.detect):
        runtime = detector()
        if runtime is not None:
            return runtime
    return cpu_backend.detect()


def get_runtime() -> RuntimeInfo:
    """Return the preferred runtime.

    Detection is intentionally re-evaluated on each call so tests and process
    supervisors can change environment or hardware visibility before model
    loading without having to clear a global cache.
    """
    return select_runtime()


def runtime_status() -> dict[str, Any]:
    """Return runtime selection details for diagnostics and dashboards."""
    return get_runtime().to_dict()
