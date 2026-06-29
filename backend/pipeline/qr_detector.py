"""QR code and barcode detection for moderation pipeline.

QR codes in content submissions are a near-certain promotional signal:
  - URLs to external products / courses
  - Payment QR codes (PhonePe, GPay, UPI)
  - Telegram / WhatsApp invite links
  - Instagram / YouTube profile QRs

Requires: apt-get install -y libzbar0 && pip install pyzbar

Falls back gracefully if pyzbar is not installed — returns score 0.0.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_URL_RE          = re.compile(r"https?://|www\.", re.IGNORECASE)
_SOCIAL_RE       = re.compile(
    r"(t\.me|wa\.me|instagram\.com|youtube\.com|facebook\.com|twitter\.com"
    r"|x\.com|whatsapp\.com|tiktok\.com|snapchat\.com|linkedin\.com)",
    re.IGNORECASE,
)
_PAYMENT_RE      = re.compile(
    r"(upi://|phonepe|gpay|paytm|razorpay|upi|bhim)",
    re.IGNORECASE,
)
_TELEGRAM_RE     = re.compile(r"t\.me/", re.IGNORECASE)


def _try_pyzbar(img: np.ndarray) -> list[str]:
    try:
        from pyzbar import pyzbar
        codes = pyzbar.decode(img)
        return [c.data.decode("utf-8", errors="replace") for c in codes if c.data]
    except ImportError:
        logger.debug("pyzbar not installed — QR detection disabled")
        return []
    except Exception:
        logger.debug("pyzbar decode failed", exc_info=True)
        return []


def _try_cv2_qr(img: np.ndarray) -> list[str]:
    """OpenCV built-in QR detector (no extra deps, lower accuracy)."""
    try:
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(img)
        return [data] if data else []
    except Exception:
        return []


def _decode_all(image_path: str) -> list[str]:
    """Try both decoders; return combined unique results."""
    img = cv2.imread(image_path)
    if img is None:
        return []

    results: list[str] = []

    # Attempt 1: pyzbar on original
    results.extend(_try_pyzbar(img))

    # Attempt 2: cv2 built-in on original
    if not results:
        results.extend(_try_cv2_qr(img))

    # Attempt 3: grayscale + threshold (improves detection on colourful backgrounds)
    if not results:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results.extend(_try_pyzbar(thresh))
        if not results:
            results.extend(_try_cv2_qr(thresh))

    return list(dict.fromkeys(r for r in results if r.strip()))


def analyze_qr(image_path: str) -> dict[str, Any]:
    """Detect QR codes and return a scored signal dict.

    Keys:
      qr_code_score      float [0, 1] — promotional risk of QR presence
      qr_code_detected   bool
      qr_decoded_text    str  — joined decoded payloads (first 3)
      qr_contains_url    bool
      qr_contains_social bool
      qr_contains_payment bool
    """
    decoded = _decode_all(image_path)

    if not decoded:
        return {
            "qr_code_score":       0.0,
            "qr_code_detected":    False,
            "qr_decoded_text":     "",
            "qr_contains_url":     False,
            "qr_contains_social":  False,
            "qr_contains_payment": False,
        }

    combined = " ".join(decoded).lower()
    contains_social   = bool(_SOCIAL_RE.search(combined))
    contains_payment  = bool(_PAYMENT_RE.search(combined))
    contains_url      = bool(_URL_RE.search(combined))
    contains_telegram = bool(_TELEGRAM_RE.search(combined))

    # Score by severity:  social link > payment > generic URL > unknown content
    if contains_telegram or contains_social:
        score = 0.95
    elif contains_payment:
        score = 0.85
    elif contains_url:
        score = 0.80
    else:
        score = 0.55   # QR code present but couldn't classify — still suspicious

    logger.info(
        "QR codes detected: count=%d url=%s social=%s payment=%s score=%.2f",
        len(decoded), contains_url, contains_social, contains_payment, score,
    )

    return {
        "qr_code_score":       score,
        "qr_code_detected":    True,
        "qr_decoded_text":     " | ".join(decoded[:3]),
        "qr_contains_url":     contains_url,
        "qr_contains_social":  contains_social,
        "qr_contains_payment": contains_payment,
    }
