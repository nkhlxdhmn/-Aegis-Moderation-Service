"""Download and verify all AI model weights required by Aegis Moderation.

Run once before first use, or after a clean install:
    python scripts/setup_models.py

Models are downloaded by their respective libraries (HuggingFace Hub,
Ultralytics) and cached in the directories pointed to by HF_HOME /
YOLO_CONFIG_DIR environment variables.  No weights are committed to git.

Exit code 0 — all models downloaded successfully.
Exit code 1 — one or more downloads failed.
"""

from __future__ import annotations

import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("setup_models")

_PASS = "PASS"
_FAIL = "FAIL"
_WARN = "WARN"
_results: list[tuple[str, str, str]] = []


def _record(label: str, status: str, detail: str = "") -> None:
    _results.append((label, status, detail))
    marker = {_PASS: "✓", _FAIL: "✗", _WARN: "!"}.get(status, "?")
    print(f"  [{marker}] {label:<50} {status}  {detail}")


def _section(title: str) -> None:
    print(f"\n── {title} {'─' * max(0, 56 - len(title))}")


def _timed(fn, label: str, fallback_status: str = _WARN) -> bool:
    t0 = time.time()
    try:
        fn()
        _record(label, _PASS, f"{time.time()-t0:.1f}s")
        return True
    except Exception as exc:
        _record(label, fallback_status, f"{exc!s:.100}")
        return False


def download_nsfw() -> None:
    _section("NSFW Classifier (Falconsai/nsfw_image_detection)")
    from backend.pipeline import nsfw
    _timed(nsfw._get_state, "NSFW model download", _FAIL)


def download_yolo() -> None:
    _section("Object Detector (YOLO11x)")
    from backend.pipeline import object_detector
    _timed(object_detector._get_state, "YOLO11x download", _FAIL)


def download_siglip() -> None:
    _section("Vision Encoder (google/siglip2-large-patch16-384)")
    from backend.pipeline import clip_engine
    _timed(clip_engine._get_state, "SigLIP2 download", _FAIL)


def download_ocr() -> None:
    _section("OCR Engines (Surya + EasyOCR)")
    from backend.pipeline.surya_ocr import load_surya
    _timed(load_surya, "Surya OCR download", _WARN)
    from backend.pipeline.easyocr_engine import load_easyocr
    _timed(load_easyocr, "EasyOCR download", _FAIL)


def download_blip() -> None:
    _section("Image Captioning (Salesforce/blip-image-captioning-large)")
    from backend.pipeline import vlm_engine
    _timed(vlm_engine._get_blip, "BLIP download", _FAIL)


def download_detoxify() -> None:
    _section("Toxicity Classifier (Detoxify multilingual)")
    from backend.pipeline.text_moderation import _get_detoxify
    _timed(_get_detoxify, "Detoxify download", _WARN)


def check_faiss() -> None:
    _section("FAISS Embedding Cache")
    try:
        import faiss  # noqa: F401
        _record("faiss-cpu import", _PASS)
    except ImportError:
        _record("faiss-cpu import", _WARN, "not installed — similarity cache disabled")


def _print_summary() -> int:
    fails = [r for r in _results if r[1] == _FAIL]
    warns = [r for r in _results if r[1] == _WARN]
    passes = [r for r in _results if r[1] == _PASS]
    print("\n" + "═" * 70)
    print(f"  SUMMARY: {len(passes)} downloaded  {len(warns)} warnings  {len(fails)} failed")
    print("═" * 70)
    if fails:
        print("\n  FAILED downloads:")
        for label, _, detail in fails:
            print(f"    • {label}: {detail}")
    if warns:
        print("\n  Warnings (non-fatal):")
        for label, _, detail in warns:
            print(f"    • {label}: {detail}")
    if not fails:
        print("\n  All required model weights are ready.\n")
        return 0
    print("\n  One or more required downloads failed — check your internet connection.\n")
    return 1


if __name__ == "__main__":
    print("═" * 70)
    print("  Aegis Moderation — Model Setup")
    print("═" * 70)
    download_nsfw()
    download_yolo()
    download_siglip()
    download_ocr()
    download_blip()
    download_detoxify()
    check_faiss()
    sys.exit(_print_summary())
