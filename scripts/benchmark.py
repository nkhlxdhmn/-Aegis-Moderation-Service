"""Benchmark OCR and end-to-end moderation latency."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Callable


def _measure(fn: Callable[[], object], iterations: int) -> list[float]:
    timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - started)
    return timings


def _summary(name: str, timings: list[float]) -> str:
    if not timings:
        return f"{name}: no samples"
    return (
        f"{name}: n={len(timings)} "
        f"avg={statistics.mean(timings):.3f}s "
        f"p50={statistics.median(timings):.3f}s "
        f"max={max(timings):.3f}s "
        f"throughput={len(timings) / sum(timings):.2f}/s"
    )


def main() -> None:
    """Run local benchmark scenarios."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Image to benchmark")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--ocr-only", action="store_true")
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    from pipeline.ocr import extract_ocr_text

    ocr_timings = _measure(lambda: extract_ocr_text(str(args.image)), args.iterations)
    print(_summary("OCR latency", ocr_timings))

    if args.ocr_only:
        return

    from pipeline.safety_flags import analyze_image

    e2e_timings = _measure(lambda: analyze_image(str(args.image), None), args.iterations)
    print(_summary("End-to-end latency", e2e_timings))
    print("CPU/GPU utilization: sample with nvidia-smi, psutil, or Prometheus exporters.")


if __name__ == "__main__":
    main()
