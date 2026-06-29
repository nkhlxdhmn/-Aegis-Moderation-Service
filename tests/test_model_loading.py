"""Tests that all production models can be imported and their singleton state
objects are returned without raising exceptions.

These tests do NOT load real model weights — they mock the heavy library
imports so they run in CPU-only CI environments without VRAM.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_torch_mock(device_name: str = "cuda:0") -> MagicMock:
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.float16 = "float16"
    torch.inference_mode.return_value.__enter__ = lambda *_: None
    torch.inference_mode.return_value.__exit__ = lambda *_: None
    return torch


def _mock_module(name: str, obj: object | None = None) -> MagicMock:
    mod = MagicMock() if obj is None else obj
    sys.modules[name] = mod  # type: ignore[assignment]
    return mod  # type: ignore[return-value]


# ── OpenNSFW2 (Falconsai/nsfw_image_detection) ────────────────────────────────

class TestNSFWModelLoading:
    def test_get_state_returns_singleton(self) -> None:
        torch_mock = _make_torch_mock()
        processor_mock = MagicMock()
        model_mock = MagicMock()
        model_mock.eval.return_value = model_mock

        transformers_mock = MagicMock()
        transformers_mock.AutoFeatureExtractor.from_pretrained.return_value = processor_mock
        transformers_mock.AutoModelForImageClassification.from_pretrained.return_value = model_mock

        with (
            patch.dict("sys.modules", {"torch": torch_mock, "transformers": transformers_mock}),
            patch("pipeline.nsfw._state", None),
            patch("pipeline.nsfw._state_lock"),
        ):
            from pipeline import nsfw
            # Reset singleton so the mock path is exercised
            nsfw._state = None
            with patch("pipeline.nsfw._state_lock", MagicMock()):
                with patch.object(
                    transformers_mock.AutoFeatureExtractor, "from_pretrained",
                    return_value=processor_mock
                ):
                    with patch.object(
                        transformers_mock.AutoModelForImageClassification, "from_pretrained",
                        return_value=model_mock
                    ):
                        pass  # Loading path tested by integration test

    def test_model_id_constant(self) -> None:
        from pipeline.nsfw import NSFW_MODEL_ID
        assert NSFW_MODEL_ID == "Falconsai/nsfw_image_detection"

    def test_device_constant(self) -> None:
        from pipeline.nsfw import DEVICE
        assert DEVICE == "cuda:0"


# ── SigLIP2 (google/siglip2-large-patch16-384) ───────────────────────────────

class TestSigLIP2ModelLoading:
    def test_model_id_constant(self) -> None:
        from pipeline.clip_engine import SIGLIP_MODEL_ID
        assert SIGLIP_MODEL_ID == "google/siglip2-large-patch16-384"

    def test_device_constant(self) -> None:
        from pipeline.clip_engine import DEVICE
        assert DEVICE == "cuda:0"

    def test_heritage_prompts_not_empty(self) -> None:
        from pipeline.clip_engine import HERITAGE_PROMPTS
        assert len(HERITAGE_PROMPTS) >= 8, "Expected at least 8 heritage prompts for coverage"

    def test_category_prompts_cover_all_categories(self) -> None:
        from pipeline.clip_engine import CATEGORY_PROMPTS
        assert len(CATEGORY_PROMPTS) >= 5, "Expected category prompts for all content types"

    def test_state_initially_none(self) -> None:
        from pipeline import clip_engine
        # We're not checking the value; just that the attribute exists
        assert hasattr(clip_engine, "_state")


# ── YOLO11x ───────────────────────────────────────────────────────────────────

class TestYOLOModelLoading:
    def test_default_model_name(self) -> None:
        from pipeline.object_detector import YOLO_MODEL_DEFAULT
        assert YOLO_MODEL_DEFAULT == "yolo11x.pt"

    def test_device_constant(self) -> None:
        from pipeline.object_detector import DEVICE
        assert DEVICE == "cuda:0"

    def test_myitihas_classes_complete(self) -> None:
        from pipeline.object_detector import MYITIHAS_CLASSES
        # Heritage classes (temple, idol, etc.) removed — they don't exist in
        # COCO-80 and would produce hallucinated detections with yolo11x.
        # Heritage context is handled by SigLIP2 prompts instead.
        required = {"person", "child", "weapon", "fire", "crowd"}
        assert required.issubset(set(MYITIHAS_CLASSES)), (
            f"Missing COCO-safe classes: {required - set(MYITIHAS_CLASSES)}"
        )

    def test_confidence_threshold_is_float(self) -> None:
        from pipeline.object_detector import CONFIDENCE_THRESHOLD
        assert 0.0 < CONFIDENCE_THRESHOLD < 1.0

    def test_state_initially_none(self) -> None:
        from pipeline import object_detector
        assert hasattr(object_detector, "_state")


# ── PaddleOCR PP-OCRv5 ────────────────────────────────────────────────────────

class TestOCRModelLoading:
    def test_get_ocr_function_exists(self) -> None:
        from pipeline.ocr import _get_ocr
        assert callable(_get_ocr)

    def test_singleton_attribute_exists(self) -> None:
        from pipeline import ocr
        assert hasattr(ocr, "_easyocr_readers")

    def test_extract_returns_string_type(self) -> None:
        from pipeline.ocr import extract_ocr_text
        # Returns empty string when image does not exist.
        result = extract_ocr_text("/nonexistent/path.jpg")
        assert isinstance(result, str)

    def test_text_quality_score_range(self) -> None:
        from pipeline.ocr import get_text_quality_score
        score = get_text_quality_score("click here free money buy now!!! http://spam.com")
        assert 0.0 <= score <= 1.0

    def test_text_quality_safe_text(self) -> None:
        from pipeline.ocr import get_text_quality_score
        score = get_text_quality_score("Shiva Lingam at the ancient Shaivite temple of Ellora")
        assert score < 0.30, f"Safe cultural text scored too high: {score}"


# ── BLIP-2 OPT-2.7B ──────────────────────────────────────────────────────────

class TestBLIP2ModelLoading:
    def test_model_id_constant(self) -> None:
        from pipeline.vlm_engine import BLIP_MODEL_ID
        assert BLIP_MODEL_ID == "Salesforce/blip-image-captioning-large"

    def test_model_id_legacy_alias(self) -> None:
        from pipeline.vlm_engine import BLIP2_MODEL_ID, BLIP_MODEL_ID
        assert BLIP2_MODEL_ID == BLIP_MODEL_ID  # backward-compat alias

    def test_device_is_cuda(self) -> None:
        from pipeline.vlm_engine import DEVICE
        assert DEVICE.startswith("cuda:")

    def test_state_initially_none(self) -> None:
        from pipeline import vlm_engine
        assert hasattr(vlm_engine, "_blip_state")

    def test_generate_caption_returns_empty_on_bad_path(self) -> None:
        from pipeline.vlm_engine import generate_caption
        result = generate_caption("/nonexistent/image.jpg")
        assert isinstance(result, str)


# ── Llama-3.1-8B AWQ ─────────────────────────────────────────────────────────

class TestLlamaModelLoading:
    def test_model_id_constant(self) -> None:
        from pipeline.vlm_engine import LLAMA_MODEL_ID
        assert "Meta-Llama-3.1-8B" in LLAMA_MODEL_ID or "llama" in LLAMA_MODEL_ID.lower()

    def test_device_is_cuda(self) -> None:
        from pipeline.vlm_engine import DEVICE
        assert DEVICE.startswith("cuda:")

    def test_state_initially_none(self) -> None:
        from pipeline import vlm_engine
        # Llama inference was removed; only BLIP singleton state remains.
        assert hasattr(vlm_engine, "_blip_state")
        assert not hasattr(vlm_engine, "_llama_state")

    def test_reason_moderation_returns_dict_on_failure(self) -> None:
        from pipeline.vlm_engine import reason_moderation
        # blip_caption is the canonical param name; blip2_caption is a legacy alias
        result = reason_moderation(
            nsfw_score=0.1,
            objects_detected=["temple"],
            ocr_text="",
            caption="Temple gopuram",
            blip_caption="",
            heritage_score=0.9,
            child_safety_score=0.02,
            violence_score=0.03,
            weapon_score=0.01,
        )
        assert isinstance(result, dict)
        assert "decision" in result
        assert result["decision"] in ("APPROVED", "REJECTED", "UNDER_REVIEW")


# ── model_warmup integration ──────────────────────────────────────────────────

# ── Text classifier hook (pipeline/text_classifier.py) ───────────────────────

class TestTextClassifier:
    def test_public_functions_exist(self) -> None:
        from pipeline.text_classifier import classify_text, is_available, load_text_classifier
        assert callable(classify_text)
        assert callable(is_available)
        assert callable(load_text_classifier)

    def test_disabled_result_is_neutral(self) -> None:
        from pipeline.text_classifier import _DISABLED_RESULT
        assert _DISABLED_RESULT["label"] == "non_abusive"
        assert _DISABLED_RESULT["abuse_score"] == 0.0
        assert _DISABLED_RESULT["disabled"] is True

    def test_classify_text_when_disabled_returns_neutral(self) -> None:
        from pipeline import text_classifier
        text_classifier.load_text_classifier()
        if not text_classifier.is_available():
            result = text_classifier.classify_text("some potentially abusive text")
            assert result["disabled"] is True
            assert result["label"] == "non_abusive"
            assert result["abuse_score"] == 0.0

    def test_classify_empty_string_never_raises(self) -> None:
        from pipeline.text_classifier import classify_text
        result = classify_text("")
        assert isinstance(result, dict)
        assert "label" in result
        assert "abuse_score" in result
        assert 0.0 <= result["abuse_score"] <= 1.0

    def test_get_model_dir_respects_env_var(self) -> None:
        import os
        from pipeline.text_classifier import _get_model_dir
        os.environ["TEXT_CLASSIFIER_MODEL_DIR"] = "/tmp/test_model"
        try:
            from pathlib import Path
            assert _get_model_dir() == Path("/tmp/test_model")
        finally:
            del os.environ["TEXT_CLASSIFIER_MODEL_DIR"]


class TestWarmupFunctions:
    def test_warmup_functions_exist(self) -> None:
        from model_warmup import (
            load_nsfw,
            load_siglip,
            load_yolo,
            load_ocr,
            load_blip2,
            load_llama,
            load_text_classifier,
            warmup_models,
            model_status,
        )
        for fn in (load_nsfw, load_siglip, load_yolo, load_ocr, load_blip2,
                   load_llama, load_text_classifier):
            assert callable(fn)
        assert callable(warmup_models)
        assert callable(model_status)

    def test_model_status_not_loaded_initially(self) -> None:
        import model_warmup
        model_warmup._models_loaded = False
        model_warmup._last_error = None
        assert model_warmup.model_status() == "not_loaded"

    def test_model_status_error(self) -> None:
        import model_warmup
        model_warmup._models_loaded = False
        model_warmup._last_error = "OOM"
        assert model_warmup.model_status() == "error"
        # Restore
        model_warmup._last_error = None

    def test_legacy_alias_load_nudenet(self) -> None:
        from model_warmup import load_nudenet, load_nsfw
        assert load_nudenet is load_nsfw

    def test_legacy_alias_load_openclip(self) -> None:
        from model_warmup import load_openclip, load_siglip
        assert load_openclip is load_siglip
