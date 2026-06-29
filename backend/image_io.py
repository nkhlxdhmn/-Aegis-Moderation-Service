"""Image ingestion helpers for uploads and public HTTPS URLs."""

from __future__ import annotations

import ipaddress
import os
import socket
import tempfile
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_SIZE_MB", "10")) * 1024 * 1024
MAX_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
REQUEST_TIMEOUT = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT_SECONDS", "12"))


class ImageInputError(ValueError):
    """Raised when a submitted image cannot be safely processed."""


def _reject_private_host(hostname: str) -> None:
    try:
        infos: Iterable[tuple] = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ImageInputError("Image URL host could not be resolved.") from exc

    for info in infos:
        address = info[4][0]
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ImageInputError("Image URL must resolve to a public internet address.")


def validate_image_url(url: str) -> str:
    """Validate an image URL before downloading it."""

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ImageInputError("Image URL must use HTTP or HTTPS.")
    if not parsed.hostname:
        raise ImageInputError("Image URL must include a host.")
    if parsed.username or parsed.password:
        raise ImageInputError("Image URL credentials are not allowed.")
    return parsed.geturl()


def _validate_image_file(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageInputError("The submitted file is not a valid image.") from exc

    if width * height > MAX_PIXELS:
        raise ImageInputError("Image dimensions are too large.")


def write_upload(contents: bytes, content_type: str | None, suffix: str = ".jpg") -> Path:
    """Persist an uploaded image to a temporary file and validate it."""

    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise ImageInputError("Only JPEG, PNG, WEBP, and GIF images are supported.")
    if not contents:
        raise ImageInputError("Image upload is empty.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise ImageInputError("Image upload exceeds the 10 MB limit.")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temp.write(contents)
        temp.close()
        path = Path(temp.name)
        _validate_image_file(path)
        return path
    except Exception:
        Path(temp.name).unlink(missing_ok=True)
        raise


def download_image(url: str) -> Path:
    """Download a public HTTPS image with SSRF and size protections."""

    current_url = validate_image_url(url)
    for _ in range(4):
        response = requests.get(
            current_url, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=False
        )
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("location")
            if not location:
                raise ImageInputError("Image URL redirect is missing a target.")
            current_url = validate_image_url(requests.compat.urljoin(current_url, location))
            continue

        response.raise_for_status()

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".img")
        total = 0
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise ImageInputError("Remote image exceeds the 10 MB limit.")
                temp.write(chunk)
            temp.close()
            path = Path(temp.name)
            _validate_image_file(path)
            return path
        except Exception:
            Path(temp.name).unlink(missing_ok=True)
            raise

    raise ImageInputError("Image URL redirects too many times.")
