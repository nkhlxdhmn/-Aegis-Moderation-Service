"""Central logging setup for the Aegis backend."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure console and rotating file logging once per process."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT)

    if not any(getattr(handler, "_aegis_console", False) for handler in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(level)
        console._aegis_console = True  # type: ignore[attr-defined]
        root.addHandler(console)

    if not any(getattr(handler, "_aegis_file", False) for handler in root.handlers):
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        file_handler._aegis_file = True  # type: ignore[attr-defined]
        root.addHandler(file_handler)
