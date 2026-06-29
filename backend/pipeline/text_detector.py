"""
text_detector.py – Smart OCR trigger.

Decides whether OCR is worth running on a given image by combining:
  - YOLO detection fast-path (certain object classes imply text presence)
  - Image-analysis heuristics (Laplacian text density, Canny edge density,
    normalised Shannon entropy over text-candidate regions)
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEXT_TRIGGER_CLASSES: frozenset[str] = frozenset(
    {"book", "cell phone", "tv", "laptop", "stop sign", "person"}
)
TEXT_DENSITY_THRESHOLD: float = 0.01  # lowered: run OCR on nearly all images
ENTROPY_THRESHOLD: float = 0.05  # lowered: catch stylised promotional text

# Number of histogram bins used for entropy estimation.
_ENTROPY_BINS: int = 32
# Pre-computed normalisation denominator so log2 isn't called at runtime.
_LOG2_BINS: float = float(np.log2(_ENTROPY_BINS))

# Laplacian threshold used for the *text_density* metric.
_LAPLACIAN_TEXT_THRESHOLD: int = 50
# Laplacian threshold used for the *text-candidate mask* (entropy region).
_LAPLACIAN_MASK_THRESHOLD: int = 30


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_text_regions(
    image_path: str,
    yolo_detections: list[dict] | None = None,
) -> tuple[bool, dict]:
    """Decide whether OCR is worth running on *image_path*.

    Parameters
    ----------
    image_path:
        Absolute or relative filesystem path to the image file.
    yolo_detections:
        Optional list of YOLO detection dicts.  Each dict must contain at
        least a ``"class"`` key with the detected object class name.  When
        any detection belongs to :data:`TEXT_TRIGGER_CLASSES` the function
        returns ``(True, metrics)`` immediately without loading the image.

    Returns
    -------
    tuple[bool, dict]
        ``(should_run_ocr, metrics)`` where *metrics* is a diagnostic dict
        containing ``text_density``, ``edge_density``, ``entropy`` (and
        optionally ``error``).
    """
    # ------------------------------------------------------------------
    # Fast-path: YOLO trigger classes
    # ------------------------------------------------------------------
    if yolo_detections:
        for det in yolo_detections:
            cls = str(det.get("class", "")).lower()
            if cls in TEXT_TRIGGER_CLASSES:
                logger.debug("OCR triggered by YOLO class '%s' for %s", cls, image_path)
                return True, {"trigger": "yolo_class", "yolo_class": cls}

    # ------------------------------------------------------------------
    # Image-analysis path
    # ------------------------------------------------------------------
    try:
        gray = _load_grayscale(image_path)
    except Exception as exc:  # noqa: BLE001
        error_msg = f"Failed to load image '{image_path}': {exc}"
        logger.warning(error_msg)
        return False, {"error": error_msg}

    try:
        text_density = _compute_text_density(gray)
        edge_density = _compute_edge_density(gray)
        entropy = _compute_entropy(gray)
    except Exception as exc:  # noqa: BLE001
        error_msg = f"Failed to compute metrics for '{image_path}': {exc}"
        logger.warning(error_msg)
        return False, {"error": error_msg}

    metrics: dict = {
        "text_density": float(text_density),
        "edge_density": float(edge_density),
        "entropy": float(entropy),
    }

    should_run_ocr = text_density > TEXT_DENSITY_THRESHOLD or entropy > ENTROPY_THRESHOLD

    logger.debug(
        "text_detector: path=%s text_density=%.4f edge_density=%.4f "
        "entropy=%.4f → should_run_ocr=%s",
        image_path,
        text_density,
        edge_density,
        entropy,
        should_run_ocr,
    )

    return should_run_ocr, metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_grayscale(image_path: str) -> np.ndarray:
    """Load *image_path* as a grayscale uint8 ndarray.

    Raises
    ------
    ValueError
        If cv2 cannot decode the file.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cv2.imread returned None for path '{image_path}'")
    return img


def _compute_text_density(gray: np.ndarray) -> float:
    """Return the fraction of pixels whose |Laplacian| exceeds the threshold."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    text_mask = np.abs(laplacian) > _LAPLACIAN_TEXT_THRESHOLD
    return float(np.mean(text_mask))


def _compute_edge_density(gray: np.ndarray) -> float:
    """Return the mean Canny-edge response normalised to [0, 1]."""
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    # edges is uint8 with values 0 or 255; dividing by 255 gives [0, 1].
    return float(np.mean(edges / 255.0))


def _compute_entropy(gray: np.ndarray) -> float:
    """Return the normalised Shannon entropy over text-candidate pixel values.

    The text-candidate region is identified by pixels whose |Laplacian|
    exceeds :data:`_LAPLACIAN_MASK_THRESHOLD`.  A 32-bin histogram of those
    pixel values is used to estimate the distribution, and the entropy is
    normalised by log2(32) so the result lies in [0, 1].
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    mask = np.abs(laplacian) > _LAPLACIAN_MASK_THRESHOLD

    candidate_pixels = gray[mask]
    if candidate_pixels.size == 0:
        return 0.0

    hist, _ = np.histogram(candidate_pixels, bins=_ENTROPY_BINS, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0

    probs = hist / total
    # Avoid log(0) by masking zero bins.
    nonzero = probs > 0
    shannon_entropy = -float(np.sum(probs[nonzero] * np.log2(probs[nonzero])))
    normalised = shannon_entropy / _LOG2_BINS
    return float(np.clip(normalised, 0.0, 1.0))
