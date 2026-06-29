"""Lightweight model validation checklist for Aegis Moderation."""

from __future__ import annotations

import sys
import time

_PASS = "PASS"
_FAIL = "FAIL"
_WARN = "WARN"
_SKIP = "SKIP"

_results: list[tuple[str, str, str]] = []


def _record(label: str, status: str, detail: str = "") -> None:
    _results.append((label, status, detail))
    marker = {"PASS": "+", "FAIL": "x", "WARN": "!", "SKIP": "-"}.get(status, "?")
    print(f"  [{marker}] {label:<45} {status}  {detail}")


def _section(title: str) -> None:
    print(f"\n-- {title} {'-' * max(0, 60 - len(title))}")


def check_gpu() -> None:
    _section("GPU / CUDA")
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
        _record("PyTorch CUDA available", _PASS if cuda_ok else _WARN, f"torch={torch.__version__}")
        if cuda_ok:
            _record("GPU count", _PASS, f"{torch.cuda.device_count()} GPU(s) visible")
        else:
            _record("GPU count", _SKIP, "CUDA not available")
    except Exception as exc:
        _record("PyTorch import", _WARN, str(exc))


def check_module_imports() -> None:
    _section("Module Imports")
    modules = (
        "backend.pipeline.nsfw",
        "backend.pipeline.clip_engine",
        "backend.pipeline.object_detector",
        "backend.pipeline.surya_ocr",
        "backend.pipeline.ocr",
        "backend.pipeline.vlm_engine",
        "backend.pipeline.text_classifier",
    )
    for module_name in modules:
        try:
            __import__(module_name)
            _record(module_name, _PASS, "imported")
        except Exception as exc:
            _record(module_name, _FAIL, str(exc)[:120])


def check_surya() -> None:
    _section("Surya OCR")
    try:
        from backend.pipeline.surya_ocr import _predictor, is_available, load_surya

        load_surya()
        if is_available():
            _record("Surya predictor", _PASS, type(_predictor).__name__)
        else:
            _record("Surya predictor", _WARN, "not loaded - surya package absent")
    except Exception as exc:
        _record("Surya initialisation", _WARN, str(exc)[:120])


def check_status_api_contract() -> None:
    _section("Model Status Contract")
    try:
        from backend.model_warmup import MODEL_KEYS, model_status_detail

        status = model_status_detail()
        _record("status keys", _PASS if set(status) == set(MODEL_KEYS) else _FAIL, str(status))
    except Exception as exc:
        _record("model status", _FAIL, str(exc)[:120])


def check_warmup() -> None:
    _section("Safe Warmup Cycle")
    try:
        from backend.model_warmup import warmup_models

        started = time.time()
        status = warmup_models()
        elapsed = time.time() - started
        _record("warmup_models()", _PASS, f"elapsed={elapsed:.1f}s status={status}")
    except Exception as exc:
        _record("warmup_models()", _FAIL, str(exc)[:120])


def _print_summary() -> int:
    fails = [r for r in _results if r[1] == _FAIL]
    warns = [r for r in _results if r[1] == _WARN]
    passes = [r for r in _results if r[1] == _PASS]
    skips = [r for r in _results if r[1] == _SKIP]

    print("\n" + "=" * 70)
    print(
        f"  SUMMARY: {len(passes)} passed  {len(warns)} warnings  "
        f"{len(fails)} failed  {len(skips)} skipped"
    )
    print("=" * 70)
    return 1 if fails else 0


if __name__ == "__main__":
    print("=" * 70)
    print("  Aegis Moderation - Model Validation Checklist")
    print("=" * 70)

    skip_warmup = "--skip-warmup" in sys.argv

    check_gpu()
    check_module_imports()
    check_surya()
    check_status_api_contract()
    if skip_warmup:
        _section("Safe Warmup Cycle")
        _record("warmup_models()", _SKIP, "--skip-warmup flag set")
    else:
        check_warmup()

    sys.exit(_print_summary())
