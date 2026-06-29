"""Image quality gate — runs before the moderation pipeline.

Rejects images that are too small, blurry, or corrupted so expensive model
inference is never wasted on unprocessable inputs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MIN_DIMENSION = 32  # pixels — reject only obviously broken/icon images; models resize internally
BLUR_THRESHOLD = 40.0  # cv2.Laplacian variance below this → blurry


def check_image_quality(image_path: str) -> tuple[bool, str | None]:
    """Return (is_ok, reason_code) where reason_code is None when OK.

    reason_code values:
        LOW_RES  — width or height < 224
        BLUR     — Laplacian variance < 40 (uniform/blurry)
        CORRUPT  — cannot be opened by PIL
    """
    logger.info("Image quality check started")
    try:
        import cv2
        import numpy as np
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
            if width < MIN_DIMENSION or height < MIN_DIMENSION:
                logger.info("Image rejected — LOW_RES (%dx%d)", width, height)
                return False, "LOW_RES"

            # Convert to greyscale numpy array for blur detection
            grey = np.array(img.convert("L"))

        laplacian_var = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        if laplacian_var < BLUR_THRESHOLD:
            logger.info("Image rejected — BLUR (var=%.1f)", laplacian_var)
            return False, "BLUR"

    except FileNotFoundError:
        logger.warning("Image quality check: file not found: %s", image_path)
        return False, "CORRUPT"
    except Exception:
        logger.exception("Image quality check failed; treating as CORRUPT")
        return False, "CORRUPT"

    logger.info("Image quality check passed")
    return True, None
