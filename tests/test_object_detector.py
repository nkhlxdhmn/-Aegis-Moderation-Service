"""Unit tests for YOLO11x object detection."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, call, patch

from backend.pipeline import object_detector


class _TensorScalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


def _make_torch_mock(*, cuda_available: bool) -> SimpleNamespace:
    """Build a minimal torch-compatible namespace for unit tests."""
    @contextmanager
    def inference_mode():
        yield

    cuda_ns = SimpleNamespace(
        is_available=Mock(return_value=cuda_available),
        empty_cache=Mock(),
    )
    return SimpleNamespace(
        cuda=cuda_ns,
        inference_mode=inference_mode,
    )


class ObjectDetectorTests(TestCase):
    def setUp(self) -> None:
        object_detector._state = None

    def tearDown(self) -> None:
        object_detector._state = None

    def _install_modules(self, *, cuda_available: bool, model: Mock) -> Mock:
        torch = _make_torch_mock(cuda_available=cuda_available)
        ultralytics = SimpleNamespace(YOLO=Mock(return_value=model))
        modules = {
            "torch": torch,
            "ultralytics": ultralytics,
        }
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        return ultralytics.YOLO

    def test_detect_objects_loads_yolo11x_lazily_and_reuses_singleton(self) -> None:
        boxes = SimpleNamespace(
            cls=[_TensorScalar(0)],
            conf=[_TensorScalar(0.96)],
        )
        result = SimpleNamespace(boxes=boxes, names={0: "person"})
        model = Mock(return_value=[result])
        model.to = Mock()
        yolo = self._install_modules(cuda_available=False, model=model)

        first = object_detector.detect_objects("image.jpg")
        second = object_detector.detect_objects("image.jpg")

        yolo.assert_called_once_with(object_detector.YOLO_MODEL_DEFAULT)
        model.to.assert_called_once_with("cpu")
        self.assertEqual(model.call_count, 2)
        self.assertEqual(first, [{"class": "person", "confidence": 0.96}])
        self.assertEqual(second, [{"class": "person", "confidence": 0.96}])

    def test_cuda_is_used_when_available(self) -> None:
        model = Mock(return_value=[])
        model.to = Mock()
        self._install_modules(cuda_available=True, model=model)

        detections = object_detector.detect_objects("image.jpg")

        model.to.assert_called_once_with(object_detector.DEVICE)
        model.assert_called_once_with(
            "image.jpg",
            device=object_detector.DEVICE,
            conf=object_detector.CONFIDENCE_THRESHOLD,
            iou=object_detector.IOU_THRESHOLD,
            verbose=False,
        )
        self.assertEqual(detections, [])

    def test_cuda_inference_falls_back_to_cpu(self) -> None:
        boxes = SimpleNamespace(
            cls=[_TensorScalar(2)],
            conf=[_TensorScalar(0.72)],
        )
        result = SimpleNamespace(boxes=boxes, names={2: "car"})
        model = Mock(side_effect=[RuntimeError("cuda failed"), [result]])
        model.to = Mock()
        self._install_modules(cuda_available=True, model=model)

        detections = object_detector.detect_objects("image.jpg")

        # model.to is called once during _get_state() with DEVICE
        model.to.assert_called_once_with(object_detector.DEVICE)
        self.assertEqual(model.call_args_list[0].kwargs["device"], object_detector.DEVICE)
        self.assertEqual(model.call_args_list[1].kwargs["device"], "cpu")
        self.assertEqual(detections, [{"class": "car", "confidence": 0.72}])

    def test_load_failure_raises_model_inference_error(self) -> None:
        torch = _make_torch_mock(cuda_available=False)
        ultralytics = SimpleNamespace(YOLO=Mock(side_effect=RuntimeError("load failed")))

        with patch.dict(sys.modules, {"torch": torch, "ultralytics": ultralytics}):
            with self.assertRaises(object_detector.ModelInferenceError):
                object_detector.detect_objects("image.jpg")

    def test_inference_failure_raises_model_inference_error(self) -> None:
        model = Mock(side_effect=RuntimeError("inference failed"))
        model.to = Mock()
        self._install_modules(cuda_available=False, model=model)

        with self.assertRaises(object_detector.ModelInferenceError):
            object_detector.detect_objects("image.jpg")
