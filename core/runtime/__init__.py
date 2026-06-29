"""Hardware runtime detection and backend selection."""

from core.runtime.detector import RuntimeInfo, get_runtime, runtime_status, select_runtime

__all__ = ["RuntimeInfo", "get_runtime", "runtime_status", "select_runtime"]
