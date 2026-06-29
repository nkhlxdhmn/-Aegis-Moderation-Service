"""Thread-based timeout wrapper for model inference calls.

Uses concurrent.futures.ThreadPoolExecutor to run inference in a daemon thread.
If the call does not complete within `timeout` seconds, TimeoutError is raised
and the thread is abandoned (it will eventually finish or be killed with the
worker process).

Note: Python threads cannot be hard-killed. Abandoned inference threads
continue consuming GPU memory and VRAM until the model call returns. For
truly hard timeouts, run inference in a subprocess — but that is incompatible
with shared GPU model state and is not done here.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Per-model timeouts (seconds).
# Cold-start (first call after container start) includes CUDA JIT compilation
# and can take 60-120s even on fast GPUs — these values must cover that.
# Subsequent warm calls are much faster (2-10s) but the timeout is not adjusted
# dynamically, so we keep it generous to avoid false timeouts under load.
TIMEOUTS: dict[str, float] = {
    "nsfw":             120.0,
    "siglip":           120.0,
    "yolo":              60.0,
    "ocr":               60.0,
    "blip":             120.0,
    "llama":            180.0,
    "qwen":             600.0,
    "embedding":         60.0,
    "text_classifier":   30.0,
}


def timeout_call(
    fn: Callable[[], T],
    timeout: float = 10.0,
    model_name: str = "unknown",
) -> T:
    """Execute fn() in a thread with a wall-clock timeout.

    Raises TimeoutError if fn() does not return within `timeout` seconds.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            logger.error(
                "Model '%s' inference timed out after %.1fs — thread abandoned",
                model_name, timeout,
            )
            raise TimeoutError(
                f"Model '{model_name}' did not respond within {timeout:.0f}s."
            )
    finally:
        executor.shutdown(wait=False)
