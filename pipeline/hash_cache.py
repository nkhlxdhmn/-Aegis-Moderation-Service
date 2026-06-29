"""Perceptual-hash deduplication cache for moderation decisions.

Primary backend: Redis (set REDIS_URL env var, TTL = 7 days).
Fallback: in-process LRU dict when Redis is unavailable or unconfigured.

imagehash.phash() is insensitive to minor resizes and JPEG artefacts,
so duplicate uploads do not pay inference costs.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

HASH_DISTANCE_THRESHOLD = int(os.getenv("HASH_DISTANCE_THRESHOLD", "5"))
MAX_CACHE_SIZE = int(os.getenv("HASH_CACHE_MAX_SIZE", "10000"))
_REDIS_TTL = 7 * 24 * 3600  # 7 days in seconds

# ── Redis backend ──────────────────────────────────────────────────────────────

_redis_client: Any = None
_redis_ok: bool = False
_redis_init_lock = threading.Lock()


def _get_redis() -> Any | None:
    """Return a live Redis client, or None if Redis is unavailable."""
    global _redis_client, _redis_ok
    if _redis_client is not None:
        return _redis_client if _redis_ok else None
    with _redis_init_lock:
        if _redis_client is not None:
            return _redis_client if _redis_ok else None
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            logger.info("REDIS_URL not set — using in-memory hash cache")
            _redis_ok = False
            return None
        try:
            import redis as _redis_lib

            local = _redis_lib.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            local.ping()
            _redis_client = local
            _redis_ok = True
            logger.info("Redis hash cache connected: %s", redis_url)
        except Exception:
            logger.warning("Redis unavailable — using in-memory hash cache fallback")
            _redis_ok = False
        return _redis_client if _redis_ok else None


# ── In-memory fallback ─────────────────────────────────────────────────────────

_cache: dict[str, dict[str, Any]] = {}
_cache_lock = threading.Lock()


# ── Perceptual hashing ─────────────────────────────────────────────────────────


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
        return sum(a != b for a, b in zip(b1.zfill(n), b2.zfill(n)))
    except (ValueError, TypeError):
        return 999


# ── Redis operations ───────────────────────────────────────────────────────────


def _redis_lookup(r: Any, hash_val: str) -> dict[str, Any] | None:
    key = f"imghash:{hash_val}"
    raw = r.get(key)
    if raw:
        logger.info("Redis hash cache hit (exact): %s", hash_val)
        return json.loads(raw)

    if HASH_DISTANCE_THRESHOLD > 0:
        # Fuzzy scan — practical for small Redis keyspaces (<100k keys)
        for k in r.scan_iter("imghash:*", count=500):
            stored = k.split(":", 1)[1] if ":" in k else k
            dist = _hamming_distance(hash_val, stored)
            if dist <= HASH_DISTANCE_THRESHOLD:
                raw = r.get(k)
                if raw:
                    logger.info(
                        "Redis hash cache hit (fuzzy, d=%d): %s ≈ %s",
                        dist,
                        hash_val,
                        stored,
                    )
                    return json.loads(raw)
    return None


def _redis_store(r: Any, hash_val: str, entry: dict[str, Any]) -> None:
    key = f"imghash:{hash_val}"
    r.setex(key, _REDIS_TTL, json.dumps(entry))
    logger.info("Redis hash stored: %s → %s", hash_val, entry.get("decision"))


# ── In-memory operations ───────────────────────────────────────────────────────


def _memory_lookup(hash_val: str) -> dict[str, Any] | None:
    with _cache_lock:
        if hash_val in _cache:
            logger.info("Memory hash cache hit (exact): %s", hash_val)
            return _cache[hash_val]
        if HASH_DISTANCE_THRESHOLD > 0:
            for stored, entry in _cache.items():
                dist = _hamming_distance(hash_val, stored)
                if dist <= HASH_DISTANCE_THRESHOLD:
                    logger.info(
                        "Memory hash cache hit (fuzzy, d=%d): %s ≈ %s",
                        dist,
                        hash_val,
                        stored,
                    )
                    return entry
    return None


def _memory_store(hash_val: str, entry: dict[str, Any]) -> None:
    with _cache_lock:
        if len(_cache) >= MAX_CACHE_SIZE:
            oldest = min(_cache, key=lambda k: _cache[k].get("created_at", 0))
            del _cache[oldest]
        _cache[hash_val] = entry
    logger.info("Memory hash stored: %s → %s", hash_val, entry.get("decision"))


# ── Public API ─────────────────────────────────────────────────────────────────


def lookup(image_path: str) -> dict[str, Any] | None:
    """Return a cached moderation decision for an image, or None if not cached."""
    hash_val = _compute_hash(image_path)
    if hash_val is None:
        return None
    r = _get_redis()
    if r:
        try:
            return _redis_lookup(r, hash_val)
        except Exception:
            logger.warning("Redis lookup failed — falling back to in-memory cache")
    return _memory_lookup(hash_val)


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

    r = _get_redis()
    if r:
        try:
            _redis_store(r, hash_val, entry)
            return hash_val
        except Exception:
            logger.warning("Redis store failed — falling back to in-memory cache")

    _memory_store(hash_val, entry)
    return hash_val


def cache_size() -> int:
    """Return approximate number of cached entries (Redis dbsize or in-memory len)."""
    r = _get_redis()
    if r:
        try:
            return r.dbsize()
        except Exception:
            pass
    with _cache_lock:
        return len(_cache)
