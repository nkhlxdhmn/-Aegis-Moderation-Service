"""Prometheus metrics for the moderation pipeline.

Falls back to no-op stubs if prometheus_client is not installed,
so the pipeline runs unchanged in environments without the package.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PROMETHEUS_AVAILABLE = False

try:
    from prometheus_client import Counter, Histogram

    _PROMETHEUS_AVAILABLE = True

    moderation_requests_total = Counter(
        "moderation_requests_total",
        "Total moderation requests by decision",
        ["decision"],
    )
    moderation_duration_seconds = Histogram(
        "moderation_duration_seconds",
        "End-to-end moderation pipeline latency in seconds",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0],
    )
    model_errors_total = Counter(
        "moderation_model_errors_total",
        "Model inference errors by model name",
        ["model"],
    )
    llama_skip_total = Counter(
        "moderation_llama_skip_total",
        "Llama+BLIP skipped due to high pre-LLM confidence",
    )
    ocr_skip_total = Counter(
        "moderation_ocr_skip_total",
        "OCR skipped — no text-bearing YOLO class detected",
    )
    hash_cache_hits_total = Counter(
        "moderation_hash_cache_hits_total",
        "Moderation decisions served from image hash cache",
    )

except ImportError:
    logger.debug("prometheus_client not installed — using no-op metric stubs")

    class _NoopCtx:
        def __enter__(self) -> _NoopCtx:
            return self

        def __exit__(self, *_: object) -> None:
            pass

    class _Noop:
        def labels(self, **_kw: object) -> _Noop:
            return self

        def inc(self, *_a: object, **_kw: object) -> None:
            pass

        def observe(self, *_a: object, **_kw: object) -> None:
            pass

        def time(self) -> _NoopCtx:
            return _NoopCtx()

    moderation_requests_total = _Noop()  # type: ignore[assignment]
    moderation_duration_seconds = _Noop()  # type: ignore[assignment]
    model_errors_total = _Noop()  # type: ignore[assignment]
    llama_skip_total = _Noop()  # type: ignore[assignment]
    ocr_skip_total = _Noop()  # type: ignore[assignment]
    hash_cache_hits_total = _Noop()  # type: ignore[assignment]
