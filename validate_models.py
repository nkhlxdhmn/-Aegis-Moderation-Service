"""Model validation checklist for moderation-service.

Run inside the container or local dev environment to confirm all components
load correctly before accepting production traffic.

Usage:
    python validate_models.py

Exit code 0 means all required checks passed.
Exit code 1 means at least one required check failed.
"""

from __future__ import annotations

import os
import sys
import time
import textwrap

_PASS = "PASS"
_FAIL = "FAIL"
_WARN = "WARN"
_SKIP = "SKIP"

_results: list[tuple[str, str, str]] = []   # (label, status, detail)


def _record(label: str, status: str, detail: str = "") -> None:
    _results.append((label, status, detail))
    marker = {"PASS": "✓", "FAIL": "✗", "WARN": "!", "SKIP": "-"}.get(status, "?")
    print(f"  [{marker}] {label:<45} {status}  {detail}")


def _section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 60 - len(title))}")


# ── 1. GPU availability ────────────────────────────────────────────────────────

def check_gpu() -> None:
    _section("GPU / CUDA")
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        _record("PyTorch CUDA available", _PASS if cuda_ok else _FAIL,
                f"torch={torch.__version__}")

        if cuda_ok:
            count = torch.cuda.device_count()
            _record("GPU count", _PASS if count >= 1 else _WARN,
                    f"{count} GPU(s) visible")
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                total_mb = props.total_memory // (1024 ** 2)
                _record(f"GPU {i}: {props.name}", _PASS, f"{total_mb} MB VRAM")
        else:
            _record("GPU count", _SKIP, "CUDA not available")
    except Exception as exc:
        _record("PyTorch import", _FAIL, str(exc))


# ── 2. GPU memory check ────────────────────────────────────────────────────────

def check_vram() -> None:
    _section("VRAM Headroom")
    try:
        import torch
        if not torch.cuda.is_available():
            _record("VRAM headroom (GPU 0)", _SKIP, "CUDA not available")
            return
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mb = props.total_memory // (1024 ** 2)
            free_mb, _ = torch.cuda.mem_get_info(i)
            free_mb //= (1024 ** 2)
            status = _PASS if free_mb >= 2048 else _WARN
            _record(f"GPU {i} free VRAM", status,
                    f"{free_mb} MB free / {total_mb} MB total")
    except Exception as exc:
        _record("VRAM check", _WARN, str(exc))


# ── 3. YOLO11x ────────────────────────────────────────────────────────────────

def check_yolo() -> None:
    _section("YOLO11x Object Detector")
    try:
        from pipeline import object_detector
        state = object_detector._get_state()
        _record("YOLO11x singleton", _PASS if state is not None else _FAIL,
                "loaded" if state is not None else "None returned")
    except Exception as exc:
        _record("YOLO11x load", _FAIL, str(exc)[:120])


# ── 4. OpenNSFW2 (NSFW) ───────────────────────────────────────────────────────

def check_nsfw() -> None:
    _section("OpenNSFW2 NSFW Classifier")
    try:
        from pipeline import nsfw
        state = nsfw._get_state()
        _record("OpenNSFW2 singleton", _PASS if state is not None else _FAIL,
                "loaded" if state is not None else "None returned")
    except Exception as exc:
        _record("OpenNSFW2 load", _FAIL, str(exc)[:120])


# ── 5. SigLIP2 ────────────────────────────────────────────────────────────────

def check_siglip() -> None:
    _section("SigLIP2 Embedding Model")
    try:
        from pipeline import clip_engine
        state = clip_engine._get_state()
        _record("SigLIP2 singleton", _PASS if state is not None else _FAIL,
                "loaded" if state is not None else "None returned")
    except Exception as exc:
        _record("SigLIP2 load", _FAIL, str(exc)[:120])


# ── 6. Surya OCR ──────────────────────────────────────────────────────────────

def check_surya() -> None:
    _section("Surya OCR (primary)")
    try:
        from pipeline.surya_ocr import load_surya, is_available, _predictor
        load_surya()
        if is_available():
            _record("Surya predictor", _PASS, type(_predictor).__name__)
        else:
            _record("Surya predictor", _WARN,
                    "not loaded — surya package absent, EasyOCR fallback active")
    except Exception as exc:
        _record("Surya initialisation", _WARN, str(exc)[:120])


# ── 7. EasyOCR fallback ───────────────────────────────────────────────────────

def check_easyocr() -> None:
    _section("EasyOCR Fallback (Indic)")
    try:
        from pipeline.easyocr_engine import load_easyocr, _readers
        load_easyocr()
        if _readers:
            _record("EasyOCR readers", _PASS, f"{len(_readers)} reader(s) loaded")
        else:
            _record("EasyOCR readers", _FAIL, "no readers loaded")
    except Exception as exc:
        _record("EasyOCR initialisation", _FAIL, str(exc)[:120])


# ── 8. BLIP captioning ────────────────────────────────────────────────────────

def check_blip() -> None:
    _section("BLIP Image Captioning")
    try:
        from pipeline import vlm_engine
        state = vlm_engine._get_blip()
        _record("BLIP singleton", _PASS if state is not None else _FAIL,
                f"device={state.device}" if state else "None returned")
    except Exception as exc:
        _record("BLIP load", _FAIL, str(exc)[:120])


# ── 9. MuRIL text classifier (optional) ──────────────────────────────────────

def check_text_classifier() -> None:
    _section("MuRIL Text Classifier (optional hook)")
    try:
        from pipeline import text_classifier
        text_classifier.load_text_classifier()
        if text_classifier.is_available():
            _record("MuRIL classifier", _PASS, "loaded — abuse scoring active")
        elif text_classifier._classifier_disabled:
            model_dir = text_classifier._get_model_dir()
            _record("MuRIL classifier", _WARN,
                    f"disabled — weights absent at {model_dir}")
        else:
            _record("MuRIL classifier", _WARN, "not yet loaded")
    except Exception as exc:
        _record("MuRIL classifier", _WARN, str(exc)[:120])


# ── 10. FAISS embedding cache ────────────────────────────────────────────────

def check_faiss() -> None:
    _section("FAISS Embedding Cache")
    try:
        import faiss
        _record("faiss-cpu import", _PASS, f"version={faiss.__version__}")
    except ImportError:
        _record("faiss-cpu import", _WARN,
                "not installed — embedding similarity cache disabled")
    except Exception as exc:
        _record("faiss import", _WARN, str(exc)[:120])


# ── 11. Full warmup cycle ────────────────────────────────────────────────────

def check_warmup() -> None:
    _section("Full Warmup Cycle")
    try:
        from model_warmup import warmup_models, model_status
        t0 = time.time()
        warmup_models()
        elapsed = time.time() - t0
        status_val = model_status()
        _record("warmup_models()", _PASS if status_val == "loaded" else _FAIL,
                f"status={status_val} elapsed={elapsed:.1f}s")
    except Exception as exc:
        _record("warmup_models()", _FAIL, str(exc)[:120])


# ── Summary ──────────────────────────────────────────────────────────────────

def _print_summary() -> int:
    fails = [r for r in _results if r[1] == _FAIL]
    warns = [r for r in _results if r[1] == _WARN]
    passes = [r for r in _results if r[1] == _PASS]
    skips = [r for r in _results if r[1] == _SKIP]

    print("\n" + "═" * 70)
    print(f"  SUMMARY: {len(passes)} passed  {len(warns)} warnings  "
          f"{len(fails)} failed  {len(skips)} skipped")
    print("═" * 70)

    if fails:
        print("\n  FAILED checks:")
        for label, _, detail in fails:
            print(f"    • {label}: {detail}")

    if warns:
        print("\n  WARNINGS (non-fatal):")
        for label, _, detail in warns:
            print(f"    • {label}: {detail}")

    if not fails:
        print("\n  All required checks passed — service is ready for traffic.\n")
        return 0
    else:
        print("\n  One or more required checks failed — do not deploy.\n")
        return 1


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  moderation-service — Model Validation Checklist")
    print("=" * 70)

    skip_warmup = "--skip-warmup" in sys.argv

    check_gpu()
    check_vram()
    check_yolo()
    check_nsfw()
    check_siglip()
    check_surya()
    check_easyocr()
    check_blip()
    check_text_classifier()
    check_faiss()

    if skip_warmup:
        _section("Full Warmup Cycle")
        _record("warmup_models()", _SKIP, "--skip-warmup flag set")
    else:
        check_warmup()

    sys.exit(_print_summary())
