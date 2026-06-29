"""
embedding_cache.py â€“ FAISS-based semantic similarity cache.

Uses SigLIP2 embeddings (via clip_engine) stored in a flat inner-product
index for approximate cosine-similarity lookups.  All mutations are protected
by a threading.Lock so the cache is safe to use from multiple threads.

Index and metadata are lazily loaded on first access and periodically
persisted to disk.
"""

import json
import logging
import os
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAISS_INDEX_FILE: Path = Path(os.getenv("EMBEDDING_CACHE_DIR", "/app/data")) / "embeddings.faiss"
METADATA_FILE: Path = Path(os.getenv("EMBEDDING_CACHE_DIR", "/app/data")) / "embedding_meta.json"
SIMILARITY_THRESHOLD: float = 0.93
MAX_CACHE_SIZE: int = 100_000

# Persist to disk every N additions (best-effort).
_PERSIST_INTERVAL: int = 100
# Fraction of oldest entries trimmed when the cache overflows.
_TRIM_FRACTION: float = 0.10

# ---------------------------------------------------------------------------
# Module-level state (lazy-loaded)
# ---------------------------------------------------------------------------

_index = None  # faiss.IndexFlatIP or None
_metadata: list[dict] = []
_lock: threading.Lock = threading.Lock()
_embedding_dim: int | None = None
_additions_since_persist: int = 0

# ---------------------------------------------------------------------------
# faiss import (optional dependency)
# ---------------------------------------------------------------------------

try:
    import faiss as _faiss  # type: ignore[import]

    _FAISS_AVAILABLE = True
except ImportError:
    _faiss = None  # type: ignore[assignment]
    _FAISS_AVAILABLE = False
    logger.warning(
        "faiss is not installed â€“ embedding_cache is disabled. "
        "Install faiss-cpu or faiss-gpu to enable semantic caching."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_or_create_index(dim: int):
    """Return a ``faiss.IndexFlatIP`` loaded from disk or freshly created.

    Also populates the module-level ``_metadata`` list.

    Parameters
    ----------
    dim:
        Embedding dimensionality.

    Returns
    -------
    faiss.IndexFlatIP | None
        ``None`` when faiss is not available.
    """
    global _metadata  # noqa: PLW0603

    if not _FAISS_AVAILABLE:
        logger.warning("_load_or_create_index called but faiss is not available.")
        return None

    loaded_index = None
    if FAISS_INDEX_FILE.exists():
        try:
            loaded_index = _faiss.read_index(str(FAISS_INDEX_FILE))
            logger.info(
                "Loaded FAISS index from %s (%d vectors)",
                FAISS_INDEX_FILE,
                loaded_index.ntotal,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load FAISS index from %s: %s â€“ creating new index.",
                FAISS_INDEX_FILE,
                exc,
            )
            loaded_index = None

    if loaded_index is None:
        loaded_index = _faiss.IndexFlatIP(dim)
        logger.info("Created new FAISS IndexFlatIP(dim=%d)", dim)

    # Load companion metadata.
    if METADATA_FILE.exists():
        try:
            with METADATA_FILE.open("r", encoding="utf-8") as fh:
                _metadata = json.load(fh)
            logger.debug("Loaded %d metadata entries from %s", len(_metadata), METADATA_FILE)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load metadata from %s: %s â€“ starting empty.", METADATA_FILE, exc
            )
            _metadata = []
    else:
        _metadata = []

    return loaded_index


def _ensure_index(dim: int):
    """Ensure ``_index`` is initialised for the given *dim*.

    Must be called while ``_lock`` is held.
    """
    global _index, _embedding_dim  # noqa: PLW0603

    if _index is not None:
        return

    _index = _load_or_create_index(dim)
    _embedding_dim = dim


def _persist() -> None:
    """Write the current FAISS index and metadata to disk (best-effort)."""
    if not _FAISS_AVAILABLE or _index is None:
        return

    try:
        FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        _faiss.write_index(_index, str(FAISS_INDEX_FILE))
        logger.debug("Persisted FAISS index to %s", FAISS_INDEX_FILE)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist FAISS index: %s", exc)

    try:
        METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with METADATA_FILE.open("w", encoding="utf-8") as fh:
            json.dump(_metadata, fh)
        logger.debug("Persisted %d metadata entries to %s", len(_metadata), METADATA_FILE)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist metadata: %s", exc)


def _trim_cache() -> None:
    """Remove the oldest ``_TRIM_FRACTION`` entries and rebuild the index.

    Must be called while ``_lock`` is held.
    """
    global _index, _metadata  # noqa: PLW0603

    if not _FAISS_AVAILABLE or _index is None:
        return

    n_remove = max(1, int(len(_metadata) * _TRIM_FRACTION))
    logger.info(
        "Cache exceeded MAX_CACHE_SIZE (%d); trimming %d oldest entries.",
        MAX_CACHE_SIZE,
        n_remove,
    )

    _metadata = _metadata[n_remove:]
    new_index = _faiss.IndexFlatIP(_embedding_dim)

    if _index.ntotal > n_remove:
        # Reconstruct all kept vectors from the old index.
        # IndexFlatIP supports reconstruct_n.
        try:
            kept_vectors = np.empty((len(_metadata), _embedding_dim), dtype=np.float32)
            for i in range(len(_metadata)):
                kept_vectors[i] = _index.reconstruct(n_remove + i)
            new_index.add(kept_vectors)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to rebuild FAISS index after trim: %s", exc)

    _index = new_index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_image_embedding(image_path: str) -> "np.ndarray | None":
    """Compute a normalised float32 embedding for *image_path*.

    Delegates to :func:`clip_engine.get_image_embedding`.

    Returns
    -------
    np.ndarray | None
        Shape ``(dim,)``, dtype float32, unit-length vector.
        ``None`` on any failure.
    """
    try:
        from backend.pipeline import clip_engine  # type: ignore[import]

        embedding = clip_engine.get_image_embedding(image_path)
        if embedding is None:
            return None
        emb = np.array(embedding, dtype=np.float32)
        # Ensure unit length.
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        return emb
    except Exception as exc:  # noqa: BLE001
        logger.error("get_image_embedding failed for '%s': %s", image_path, exc)
        return None


def search_similar_image(
    image_path: str,
    embedding: "np.ndarray | None" = None,
) -> "dict | None":
    """Look up the most similar cached result for *image_path*.

    Parameters
    ----------
    image_path:
        Path to the query image (used to compute the embedding when
        *embedding* is not supplied).
    embedding:
        Pre-computed embedding array.  If ``None`` it is computed on-the-fly.

    Returns
    -------
    dict | None
        Metadata dict of the nearest cached image with an added
        ``"similarity"`` key, or ``None`` when no match exceeds
        :data:`SIMILARITY_THRESHOLD`.
    """
    if not _FAISS_AVAILABLE:
        return None

    if embedding is None:
        embedding = get_image_embedding(image_path)
    if embedding is None:
        return None

    dim = embedding.shape[0]

    with _lock:
        _ensure_index(dim)
        if _index is None or _index.ntotal == 0:
            return None

        query = embedding.reshape(1, -1).astype(np.float32)
        try:
            D, indices = _index.search(query, k=1)
        except Exception as exc:  # noqa: BLE001
            logger.error("FAISS search failed: %s", exc)
            return None

        similarity = float(D[0][0])
        idx = int(indices[0][0])

        if similarity >= SIMILARITY_THRESHOLD and idx >= 0 and idx < len(_metadata):
            result = dict(_metadata[idx])
            result["similarity"] = similarity
            logger.debug("Cache hit for '%s': similarity=%.4f idx=%d", image_path, similarity, idx)
            return result

    logger.debug(
        "No cache hit for '%s': best_similarity=%.4f (threshold=%.2f)",
        image_path,
        similarity,
        SIMILARITY_THRESHOLD,
    )
    return None


def store_image(
    image_path: str,
    embedding: "np.ndarray | None",
    decision: str,
    scores: dict,
    evidence: dict | None = None,
) -> bool:
    """Add an image embedding and its moderation result to the cache.

    Parameters
    ----------
    image_path:
        Path used as a record identifier in metadata.
    embedding:
        Float32 embedding array (will be L2-normalised before storage).
    decision:
        Moderation decision string (e.g. ``"APPROVED"``, ``"REJECTED"``).
    scores:
        Risk score dict to store alongside the decision.
    evidence:
        Optional additional evidence dict.

    Returns
    -------
    bool
        ``True`` on success, ``False`` otherwise.
    """
    global _additions_since_persist  # noqa: PLW0603

    if not _FAISS_AVAILABLE:
        return False
    if embedding is None:
        return False

    emb_copy = embedding.copy().astype(np.float32).reshape(1, -1)
    _faiss.normalize_L2(emb_copy)
    dim = emb_copy.shape[1]

    with _lock:
        _ensure_index(dim)
        if _index is None:
            return False

        try:
            _index.add(emb_copy)
        except Exception as exc:  # noqa: BLE001
            logger.error("FAISS add failed: %s", exc)
            return False

        _metadata.append(
            {
                "decision": decision,
                "scores": scores,
                "evidence": evidence or {},
                "image_path": image_path,
            }
        )

        _additions_since_persist += 1

        if _additions_since_persist >= _PERSIST_INTERVAL:
            _persist()
            _additions_since_persist = 0

        if len(_metadata) > MAX_CACHE_SIZE:
            _trim_cache()
            # Persist after trim so the pruned state survives a restart.
            _persist()

    logger.debug(
        "Stored embedding for '%s' â†’ decision=%s (cache_size=%d)",
        image_path,
        decision,
        len(_metadata),
    )
    return True
