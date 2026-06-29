"""Per-model circuit breaker for moderation pipeline resilience.

After max_failures consecutive failures, the breaker opens for cooldown_seconds.
During the open state, calls are rejected immediately with CircuitOpenError.
The breaker resets automatically after the cooldown expires.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised when a circuit breaker is open (model temporarily disabled)."""


class CircuitBreaker:
    """Thread-safe circuit breaker with automatic cooldown reset."""

    def __init__(
        self,
        name: str,
        max_failures: int = 5,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.name = name
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds

        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._is_open_locked()

    def _is_open_locked(self) -> bool:
        if self._opened_at is None:
            return False
        elapsed = time.monotonic() - self._opened_at
        if elapsed >= self.cooldown_seconds:
            # Auto-recover after cooldown
            self._failures = 0
            self._opened_at = None
            logger.info("CircuitBreaker '%s' recovered after cooldown", self.name)
            return False
        return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            logger.warning(
                "CircuitBreaker '%s': failure %d/%d",
                self.name, self._failures, self.max_failures,
            )
            if self._failures >= self.max_failures and self._opened_at is None:
                self._opened_at = time.monotonic()
                logger.error(
                    "CircuitBreaker '%s' OPENED — model disabled for %.0fs",
                    self.name, self.cooldown_seconds,
                )

    def call(self, fn: Callable[[], T], fallback: T | None = None) -> T:
        """Call fn, recording success/failure. Raises CircuitOpenError if open."""
        with self._lock:
            if self._is_open_locked():
                raise CircuitOpenError(
                    f"Model '{self.name}' is temporarily disabled "
                    f"({self._failures} consecutive failures)."
                )
        try:
            result = fn()
            self.record_success()
            return result
        except CircuitOpenError:
            raise
        except Exception:
            self.record_failure()
            raise

    def status(self) -> dict[str, Any]:
        with self._lock:
            open_state = self._is_open_locked()
            remaining = 0.0
            if open_state and self._opened_at is not None:
                remaining = max(
                    0.0,
                    self.cooldown_seconds - (time.monotonic() - self._opened_at),
                )
            return {
                "name": self.name,
                "state": "open" if open_state else "closed",
                "failures": self._failures,
                "cooldown_remaining_seconds": round(remaining, 1),
            }


# ── Per-model circuit breakers ─────────────────────────────────────────────────
nsfw_breaker = CircuitBreaker("nsfw", max_failures=5, cooldown_seconds=60)
siglip_breaker = CircuitBreaker("siglip", max_failures=5, cooldown_seconds=60)
yolo_breaker = CircuitBreaker("yolo", max_failures=5, cooldown_seconds=60)
ocr_breaker = CircuitBreaker("ocr", max_failures=5, cooldown_seconds=60)
blip_breaker = CircuitBreaker("blip", max_failures=5, cooldown_seconds=60)
llama_breaker = CircuitBreaker("llama", max_failures=5, cooldown_seconds=60)
qwen_breaker = CircuitBreaker("qwen", max_failures=5, cooldown_seconds=90)
embedding_breaker  = CircuitBreaker("embedding",  max_failures=5, cooldown_seconds=60)
toxicity_breaker          = CircuitBreaker("toxicity",          max_failures=3, cooldown_seconds=120)
text_classifier_breaker   = CircuitBreaker("text_classifier",   max_failures=3, cooldown_seconds=120)


def all_statuses() -> list[dict[str, Any]]:
    return [b.status() for b in (
        nsfw_breaker, siglip_breaker, yolo_breaker,
        ocr_breaker, blip_breaker, llama_breaker,
        qwen_breaker, embedding_breaker, toxicity_breaker,
        text_classifier_breaker,
    )]
