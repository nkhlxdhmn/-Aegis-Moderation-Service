from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.runtime import get_runtime, runtime_status


def _torch_cuda_available() -> SimpleNamespace:
    cuda = SimpleNamespace(
        is_available=Mock(return_value=True),
        device_count=Mock(return_value=1),
        get_device_name=Mock(return_value="NVIDIA Test GPU"),
    )
    return SimpleNamespace(cuda=cuda)


def _torch_cuda_unavailable() -> SimpleNamespace:
    cuda = SimpleNamespace(is_available=Mock(return_value=False))
    return SimpleNamespace(cuda=cuda)


def _ort_with(*providers: str) -> SimpleNamespace:
    return SimpleNamespace(get_available_providers=Mock(return_value=list(providers)))


def test_cuda_has_priority_over_npu() -> None:
    modules = {
        "torch": _torch_cuda_available(),
        "onnxruntime": _ort_with("VitisAIExecutionProvider", "CPUExecutionProvider"),
    }

    with patch.dict(sys.modules, modules):
        runtime = get_runtime()

    assert runtime.name == "cuda"
    assert runtime.torch_device == "cuda:0"
    assert runtime.provider == "CUDAExecutionProvider"


def test_npu_selected_when_cuda_is_unavailable() -> None:
    modules = {
        "torch": _torch_cuda_unavailable(),
        "onnxruntime": _ort_with("VitisAIExecutionProvider", "CPUExecutionProvider"),
    }

    with patch.dict(sys.modules, modules):
        runtime = get_runtime()

    assert runtime.name == "npu"
    assert runtime.supports_onnx is True
    assert runtime.supports_torch is False
    assert runtime.torch_device == "cpu"
    assert runtime.execution_providers == ("VitisAIExecutionProvider", "CPUExecutionProvider")


def test_cpu_fallback_when_npu_is_unavailable() -> None:
    modules = {
        "torch": _torch_cuda_unavailable(),
        "onnxruntime": _ort_with("CPUExecutionProvider"),
    }

    with patch.dict(sys.modules, modules):
        runtime = get_runtime()

    assert runtime.name == "cpu"
    assert runtime.device == "cpu"
    assert runtime.execution_providers == ("CPUExecutionProvider",)


def test_cpu_fallback_when_optional_runtime_packages_are_missing() -> None:
    with patch.dict(sys.modules, {"torch": None, "onnxruntime": None}):
        runtime = get_runtime()

    assert runtime.name == "cpu"


def test_runtime_status_is_json_serializable_shape() -> None:
    with patch.dict(sys.modules, {"torch": None, "onnxruntime": None}):
        status = runtime_status()

    assert status["name"] == "cpu"
    assert status["torch_device"] == "cpu"
    assert isinstance(status["execution_providers"], list)
