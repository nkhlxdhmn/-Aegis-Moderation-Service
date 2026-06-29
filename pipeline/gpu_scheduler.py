"""GPU load monitoring and device recommendation.

Tracks memory utilization on GPU0 and GPU1 and recommends the
less-loaded device when a new model is about to be assigned.

Note: models already loaded cannot be migrated without reloading.
This scheduler is most useful at startup or during circuit-breaker
recovery when a model is being re-initialized.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Above this fraction of total VRAM a GPU is considered overloaded.
GPU_OVERLOAD_THRESHOLD: float = 0.85


@dataclass
class GPULoad:
    device_id: int
    name: str
    allocated_mb: float
    total_mb: float
    utilization_pct: float
    is_available: bool


def get_gpu_loads() -> list[GPULoad]:
    """Return current memory allocation for GPU0 and GPU1."""
    loads: list[GPULoad] = []
    try:
        import torch

        if not torch.cuda.is_available():
            return loads
        count = min(2, torch.cuda.device_count())
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            total = props.total_memory
            pct = (allocated / total) if total else 0.0
            loads.append(
                GPULoad(
                    device_id=i,
                    name=props.name,
                    allocated_mb=round(allocated / 1024**2, 1),
                    total_mb=round(total / 1024**2, 1),
                    utilization_pct=round(pct * 100, 1),
                    is_available=pct < GPU_OVERLOAD_THRESHOLD,
                )
            )
    except Exception:
        logger.exception("GPU load query failed")
    return loads


def recommend_device(preferred: str) -> str:
    """Return preferred device, or the alternative GPU if preferred is overloaded.

    Only swaps between cuda:0 and cuda:1.  Any other string is returned as-is.
    """
    alt = {"cuda:0": "cuda:1", "cuda:1": "cuda:0"}.get(preferred)
    if alt is None:
        return preferred

    loads = get_gpu_loads()
    if not loads:
        return preferred

    load_map = {f"cuda:{g.device_id}": g for g in loads}
    pref_load = load_map.get(preferred)
    if pref_load and not pref_load.is_available:
        alt_load = load_map.get(alt)
        if alt_load and alt_load.is_available:
            logger.warning(
                "GPU %s at %.1f%% VRAM — recommending %s for new model assignment",
                preferred,
                pref_load.utilization_pct,
                alt,
            )
            return alt
    return preferred


def flush_overloaded_gpus() -> None:
    """Empty CUDA cache and run GC on any GPU above the overload threshold."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for load in get_gpu_loads():
            if not load.is_available:
                logger.warning(
                    "GPU%d at %.1f%% VRAM — flushing cache",
                    load.device_id,
                    load.utilization_pct,
                )
                torch.cuda.empty_cache()
                gc.collect()
    except Exception:
        logger.exception("GPU flush failed")


def status_dicts() -> list[dict]:
    """Serialize GPU loads for the /model-health response."""
    return [
        {
            "gpu": g.device_id,
            "name": g.name,
            "allocated_mb": g.allocated_mb,
            "total_mb": g.total_mb,
            "utilization_pct": g.utilization_pct,
            "available": g.is_available,
        }
        for g in get_gpu_loads()
    ]
