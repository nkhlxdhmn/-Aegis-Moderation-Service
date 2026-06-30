"""Micro-batch inference for the moderation pipeline.

Processes N images concurrently (data-parallel) rather than sequentially.
Batch size is chosen dynamically by queue depth:

  queue_depth ≤ 5   → batch_size = 4
  queue_depth 6–20  → batch_size = 8
  queue_depth > 20  → batch_size = 16

Each image still goes through the full single-image pipeline (safety_flags.analyze_image)
in its own thread.  The benefit is that GPU forward passes from different images
interleave in the CUDA stream, reducing overall wall-clock time for a burst of work.

Usage:
    from backend.pipeline.batch_analyzer import analyze_batch, batch_size_for_depth

    batch = [(path1, caption1), (path2, caption2), ...]
    results = analyze_batch(batch, queue_depth=queue_service.get_depth())
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.pipeline.safety_flags import ModerationPipelineResult, analyze_image

logger = logging.getLogger(__name__)

# Maximum concurrent images per batch tier
_BATCH_SIZES = (
    (20, 8),  # queue_depth ≤ 20 → 8
    (50, 16),  # queue_depth ≤ 50 → 16
    (None, 32),  # queue_depth > 50 → 32
)
_MIN_BATCH = 4

# Shared executor for batch processing; sized to the largest batch tier
_batch_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="batch")


def batch_size_for_depth(queue_depth: int) -> int:
    """Return the recommended batch size given the current queue depth."""
    if queue_depth <= 5:
        return _MIN_BATCH
    for threshold, size in _BATCH_SIZES:
        if threshold is None or queue_depth <= threshold:
            return size
    return _BATCH_SIZES[-1][1]


def analyze_batch(
    items: Sequence[tuple[str, str | None]],
    queue_depth: int = 0,
) -> list[ModerationPipelineResult]:
    """Moderate a batch of images concurrently.

    Args:
        items: Sequence of (image_path, caption) pairs.
        queue_depth: Current worker queue depth (used to pick batch size).

    Returns:
        List of ModerationPipelineResult in the same order as items.
    """
    if not items:
        return []

    batch_sz = batch_size_for_depth(queue_depth)
    logger.info(
        "Batch moderation: %d images, queue_depth=%d, batch_size=%d",
        len(items),
        queue_depth,
        batch_sz,
    )

    results: list[ModerationPipelineResult | None] = [None] * len(items)
    futures = {}

    # Submit in sub-batches to avoid overwhelming the GPU
    for start in range(0, len(items), batch_sz):
        sub = items[start : start + batch_sz]
        for offset, (image_path, caption) in enumerate(sub):
            idx = start + offset
            future = _batch_executor.submit(analyze_image, image_path, caption)
            futures[future] = idx

    for future in as_completed(futures):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception:
            logger.exception("Batch item %d failed", idx)
            from backend.pipeline.safety_flags import ModerationPipelineResult, _default_scores

            results[idx] = ModerationPipelineResult(
                scores=_default_scores(),
                category_scores={},
                ocr_text="",
                pipeline_error=True,
                error_reason="Batch item processing failed.",
            )

    return [r for r in results if r is not None]
