"""In-process perceptual-hash deduplication cache for moderation decisions.

The standalone application intentionally uses local memory only. Duplicate uploads can
reuse recent decisions without requiring Redis, a database, or any external service.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

HASH_DISTANCE_THRESHOLD = int(os.getenv("HASH_DISTANCE_THRESHOLD", "5"))
MAX_CACHE_SIZE = int(os.getenv("HASH_CACHE_MAX_SIZE", "10000"))

_cache: dict[str, dict[str, Any]] = {}
_cache_lock = threading.Lock()


def _compute_hash(image_path: str) -> str | None:
    try:
        import imagehash
        from PIL import Image

        with Image.open(image_path) as img:
            return str(imagehash.phash(img))
    except Exception:
        logger.exception("Image hash computation failed")
        return None


def _hamming_distance(h1: str, h2: str) -> int:
    try:
        b1 = bin(int(h1, 16))
        b2 = bin(int(h2, 16))
        n = max(len(b1), len(b2))
        return sum(a != b for a, b in zip(b1.zfill(n), b2.zfill(n), strict=False))
    except (ValueError, TypeError):
        return 999


def lookup(image_path: str) -> dict[str, Any] | None:
    """Return a cached moderation decision for an image, or None if not cached."""

    hash_val = _compute_hash(image_path)
    if hash_val is None:
        return None

    with _cache_lock:
        if hash_val in _cache:
            logger.info("Memory hash cache hit (exact): %s", hash_val)
            return _cache[hash_val]
        if HASH_DISTANCE_THRESHOLD > 0:
            for stored, entry in _cache.items():
                dist = _hamming_distance(hash_val, stored)
                if dist <= HASH_DISTANCE_THRESHOLD:
                    logger.info(
                        "Memory hash cache hit (fuzzy, d=%d): %s ~= %s",
                        dist,
                        hash_val,
                        stored,
                    )
                    return entry
    return None


def store(
    image_path: str,
    decision: str,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> str | None:
    """Store a moderation decision keyed by perceptual hash. Returns the hash string."""

    hash_val = _compute_hash(image_path)
    if hash_val is None:
        return None

    entry: dict[str, Any] = {
        "image_hash": hash_val,
        "decision": decision,
        "reason": reason,
        "created_at": time.time(),
        **(extra or {}),
    }

    with _cache_lock:
        if len(_cache) >= MAX_CACHE_SIZE:
            oldest = min(_cache, key=lambda key: _cache[key].get("created_at", 0))
            del _cache[oldest]
        _cache[hash_val] = entry
    logger.info("Memory hash stored: %s -> %s", hash_val, entry.get("decision"))
    return hash_val


def cache_size() -> int:
    """Return the approximate number of cached entries."""

    with _cache_lock:
        return len(_cache)
