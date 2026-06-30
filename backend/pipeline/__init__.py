"""Moderation pipeline package for Aegis content safety decisions."""

import os

# Pin to GPU 0 before any CUDA library initialises.  docker-compose restricts
# the container to GPU 0 via device_ids; this guards against misconfigurations
# when the container is run outside compose (e.g. bare docker run).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
