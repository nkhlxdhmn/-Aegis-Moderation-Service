"""Warm up all AI model weights before the server accepts traffic.

Loads every model into VRAM so the first request is not slowed by cold-start
latency.  Call from your container entrypoint when MODEL_WARMUP=true, or run
directly:
    python scripts/warmup.py

Exit code 0 — all models loaded successfully.
Exit code 1 — one or more models failed to load.
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

if __name__ == "__main__":
    from backend.model_warmup import warmup_models

    try:
        warmup_models()
        print("Warmup complete — all models are ready.")
        sys.exit(0)
    except Exception as exc:
        print(f"Warmup failed: {exc}", file=sys.stderr)
        sys.exit(1)
