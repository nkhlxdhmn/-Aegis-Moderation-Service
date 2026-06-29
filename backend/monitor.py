"""Aegis Moderation — in-process monitoring and observability state.

Collects request, model, and system metrics into thread-safe in-memory
ring buffers so /api/v1/monitor/* endpoints return live data without
any external database.
"""

from __future__ import annotations

import collections
import logging
import os
import sys
import threading
import time
from typing import Any

APP_START_TIME = time.time()

_PSUTIL_AVAILABLE = False
try:
    import psutil as _psutil  # noqa: F401
    _PSUTIL_AVAILABLE = True
except ImportError:
    pass

_TORCH_AVAILABLE = False
try:
    import torch as _torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    pass

_MAX_REQUESTS = 500
_MAX_LOGS = 600
_MAX_ERRORS = 200
_MAX_SECURITY = 200
_MAX_INF_SAMPLES = 100


class _LogCapture(logging.Handler):
    """Appends formatted log records to a shared ring-buffer deque."""

    def __init__(self, buf: collections.deque) -> None:  # type: ignore[type-arg]
        super().__init__()
        self._buf = buf
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buf.append(
                {
                    "ts": record.created,
                    "level": record.levelname,
                    "logger": record.name,
                    "message": self.format(record),
                }
            )
        except Exception:
            pass


class AegisMonitor:
    """Thread-safe in-process monitoring state for Aegis Moderation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.start_time = time.time()

        # Ring buffers
        self._requests: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=_MAX_REQUESTS
        )
        self._errors: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=_MAX_ERRORS
        )
        self._security: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=_MAX_SECURITY
        )
        self._logs: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=_MAX_LOGS
        )

        # Model timing
        self._model_load_times: dict[str, float] = {}
        self._inference: dict[str, collections.deque[float]] = {}

        # Cumulative counters (reset on restart)
        self._total_requests = 0
        self._ok_count = 0
        self._err_count = 0
        self._content_type_count: dict[str, int] = {}
        self._decision_count: dict[str, int] = {}
        self._category_flag_count: dict[str, int] = {}

        # Attach log capture to root logger
        handler = _LogCapture(self._logs)
        handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(handler)

    # ── Write API ────────────────────────────────────────────────────────────

    def record_request(
        self,
        *,
        endpoint: str,
        status: str,
        duration: float,
        decision: str = "unknown",
        content_type: str = "",
        categories: dict[str, float] | None = None,
    ) -> None:
        cats = categories or {}
        ct = content_type or endpoint
        rec: dict[str, Any] = {
            "ts": time.time(),
            "endpoint": endpoint,
            "status": status,
            "duration": duration,
            "decision": decision,
            "content_type": ct,
            "categories": {k: v for k, v in cats.items() if v > 0},
        }
        with self._lock:
            self._requests.append(rec)
            self._total_requests += 1
            if status == "ok":
                self._ok_count += 1
            else:
                self._err_count += 1
            if ct in ("image", "text", "video", "pdf", "docx"):
                self._content_type_count[ct] = self._content_type_count.get(ct, 0) + 1
            self._decision_count[decision] = self._decision_count.get(decision, 0) + 1
            for cat, score in cats.items():
                if score > 50:
                    self._category_flag_count[cat] = (
                        self._category_flag_count.get(cat, 0) + 1
                    )

    def record_error(self, *, endpoint: str, error_type: str, detail: str) -> None:
        with self._lock:
            self._errors.append(
                {
                    "ts": time.time(),
                    "endpoint": endpoint,
                    "error_type": error_type,
                    "detail": detail[:500],
                }
            )

    def record_security_event(self, *, event_type: str, detail: str) -> None:
        with self._lock:
            self._security.append(
                {"ts": time.time(), "event_type": event_type, "detail": detail[:500]}
            )

    def record_model_load(self, model: str, duration: float) -> None:
        with self._lock:
            self._model_load_times[model] = duration

    def record_inference(self, model: str, duration: float) -> None:
        with self._lock:
            if model not in self._inference:
                self._inference[model] = collections.deque(maxlen=_MAX_INF_SAMPLES)
            self._inference[model].append(duration)

    # ── Read API ─────────────────────────────────────────────────────────────

    def get_system_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "cpu_percent": None,
            "memory": None,
            "disk": None,
            "network": None,
            "process_memory_mb": None,
            "active_threads": threading.active_count(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "gpu": None,
        }

        if _PSUTIL_AVAILABLE:
            import psutil

            try:
                stats["cpu_percent"] = psutil.cpu_percent(interval=None)
            except Exception:
                pass

            try:
                vm = psutil.virtual_memory()
                stats["memory"] = {
                    "total_gb": round(vm.total / 1e9, 2),
                    "used_gb": round(vm.used / 1e9, 2),
                    "available_gb": round(vm.available / 1e9, 2),
                    "percent": vm.percent,
                }
            except Exception:
                pass

            try:
                root = "C:\\" if sys.platform == "win32" else "/"
                du = psutil.disk_usage(root)
                stats["disk"] = {
                    "total_gb": round(du.total / 1e9, 2),
                    "used_gb": round(du.used / 1e9, 2),
                    "free_gb": round(du.free / 1e9, 2),
                    "percent": du.percent,
                }
            except Exception:
                pass

            try:
                net = psutil.net_io_counters()
                stats["network"] = {
                    "bytes_sent_mb": round(net.bytes_sent / 1e6, 2),
                    "bytes_recv_mb": round(net.bytes_recv / 1e6, 2),
                    "packets_sent": net.packets_sent,
                    "packets_recv": net.packets_recv,
                }
            except Exception:
                pass

            try:
                proc = psutil.Process(os.getpid())
                stats["process_memory_mb"] = round(proc.memory_info().rss / 1e6, 1)
            except Exception:
                pass

        if _TORCH_AVAILABLE:
            import torch

            try:
                if torch.cuda.is_available():
                    gpus = []
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        alloc = torch.cuda.memory_allocated(i)
                        reserved = torch.cuda.memory_reserved(i)
                        total = props.total_memory
                        gpus.append(
                            {
                                "index": i,
                                "name": props.name,
                                "total_gb": round(total / 1e9, 2),
                                "allocated_gb": round(alloc / 1e9, 2),
                                "reserved_gb": round(reserved / 1e9, 2),
                                "utilization_percent": round(alloc / total * 100, 1)
                                if total > 0
                                else 0,
                            }
                        )
                    stats["gpu"] = gpus or None
            except Exception:
                pass

        return stats

    def get_request_analytics(self) -> dict[str, Any]:
        with self._lock:
            reqs = list(self._requests)
            total = self._total_requests
            ok = self._ok_count
            err = self._err_count
            ct_counts = dict(self._content_type_count)

        now = time.time()
        durations = [r["duration"] for r in reqs]
        avg_dur = sum(durations) / len(durations) if durations else 0

        slowest = sorted(reqs, key=lambda r: r["duration"], reverse=True)[:5]

        # RPS over last 60 s
        recent_60 = [r for r in reqs if now - r["ts"] <= 60]
        rps = len(recent_60) / 60.0 if recent_60 else 0

        # 60-bucket chart history (10 s per bucket, newest last)
        buckets: dict[int, int] = {}
        for r in reqs:
            b = int((now - r["ts"]) // 10)
            if 0 <= b < 60:
                buckets[b] = buckets.get(b, 0) + 1
        history = [buckets.get(i, 0) for i in range(59, -1, -1)]

        return {
            "total": total,
            "successful": ok,
            "failed": err,
            "avg_response_time_s": round(avg_dur, 3),
            "requests_per_second": round(rps, 3),
            "slowest_requests": [
                {
                    "endpoint": r["endpoint"],
                    "duration_s": round(r["duration"], 3),
                    "decision": r.get("decision", ""),
                    "ts": r["ts"],
                }
                for r in slowest
            ],
            "history_10s_buckets": history,
            "by_content_type": {
                k: ct_counts.get(k, 0)
                for k in ("image", "text", "video", "pdf", "docx")
            },
        }

    def get_moderation_analytics(self) -> dict[str, Any]:
        with self._lock:
            ct = dict(self._content_type_count)
            decisions = dict(self._decision_count)
            cats = dict(self._category_flag_count)

        return {
            "by_content_type": {
                "images": ct.get("image", 0),
                "videos": ct.get("video", 0),
                "pdfs": ct.get("pdf", 0),
                "docx": ct.get("docx", 0),
                "text": ct.get("text", 0),
            },
            "decisions": decisions,
            "category_flags": cats,
        }

    def get_model_stats(self) -> dict[str, Any]:
        with self._lock:
            load_times = dict(self._model_load_times)
            inf = {k: list(v) for k, v in self._inference.items()}

        try:
            from backend.model_warmup import model_status_detail

            status = model_status_detail()
        except Exception:
            status = {}

        breakdown = []
        for model, times in inf.items():
            if times:
                breakdown.append(
                    {
                        "name": model,
                        "avg_inference_ms": round(sum(times) / len(times) * 1000, 1),
                        "total_inferences": len(times),
                        "min_ms": round(min(times) * 1000, 1),
                        "max_ms": round(max(times) * 1000, 1),
                        "load_time_s": round(load_times.get(model, 0), 2),
                    }
                )

        return {
            "model_status": status,
            "load_times": {k: round(v, 2) for k, v in load_times.items()},
            "inference_breakdown": breakdown,
        }

    def get_performance_stats(self) -> dict[str, Any]:
        with self._lock:
            reqs = list(self._requests)
            inf = {k: list(v) for k, v in self._inference.items()}

        now = time.time()
        recent_5m = [r for r in reqs if now - r["ts"] <= 300]
        throughput = len(recent_5m) / 5.0 if recent_5m else 0

        ok_durs = [r["duration"] for r in reqs if r["status"] == "ok"]
        avg_e2e = sum(ok_durs) / len(ok_durs) if ok_durs else 0

        def _avg(key: str) -> float:
            vals = inf.get(key, [])
            return round(sum(vals) / len(vals), 3) if vals else 0

        return {
            "avg_e2e_latency_s": round(avg_e2e, 3),
            "throughput_per_minute": round(throughput, 1),
            "avg_ocr_time_s": _avg("ocr"),
            "avg_vision_time_s": _avg("vision"),
            "avg_nlp_time_s": _avg("nlp"),
            "inference_by_model": {
                k: round(sum(v) / len(v), 3) if v else 0 for k, v in inf.items()
            },
        }

    def get_logs(
        self,
        *,
        level: str | None = None,
        search: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self._lock:
            logs = list(self._logs)
        if level:
            level_up = level.upper()
            logs = [line for line in logs if line["level"] == level_up]
        if search:
            s = search.lower()
            logs = [line for line in logs if s in line["message"].lower()]
        return logs[-limit:]

    def get_health(self) -> dict[str, Any]:
        system = self.get_system_stats()
        try:
            from backend.model_warmup import model_status_detail

            models = model_status_detail()
        except Exception:
            models = {}

        ocr_ok = any(v == "loaded" for k, v in models.items() if "ocr" in k)
        vision_ok = any(
            models.get(m) == "loaded" for m in ("nsfw", "siglip", "yolo", "blip")
        )
        text_ok = models.get("text_classifier") in ("loaded", "disabled")

        disk = system.get("disk") or {}
        disk_pct = disk.get("percent", 0)
        disk_status = (
            "healthy" if disk_pct < 80 else ("warning" if disk_pct < 95 else "offline")
        )

        mem = system.get("memory") or {}
        mem_pct = mem.get("percent", 0)
        mem_status = (
            "healthy" if mem_pct < 80 else ("warning" if mem_pct < 95 else "offline")
        )

        gpu_status = "healthy" if system.get("gpu") else "warning"

        def _icon(status: str) -> str:
            return {"healthy": "🟢", "warning": "🟡", "offline": "🔴"}.get(
                status, "⚪"
            )

        def _component(label: str, status: str) -> dict[str, str]:
            return {"label": label, "status": status, "icon": _icon(status)}

        return {
            "api": _component("API Server", "healthy"),
            "ocr": _component("OCR Engine", "healthy" if ocr_ok else "warning"),
            "vision_model": _component(
                "Vision Models", "healthy" if vision_ok else "warning"
            ),
            "text_model": _component(
                "Text Model", "healthy" if text_ok else "warning"
            ),
            "disk": _component("Disk Storage", disk_status),
            "gpu": _component("GPU", gpu_status),
            "memory": _component("Memory", mem_status),
        }

    def get_errors(self) -> dict[str, Any]:
        with self._lock:
            errors = list(self._errors)
        by_type: dict[str, int] = {}
        for e in errors:
            by_type[e["error_type"]] = by_type.get(e["error_type"], 0) + 1
        return {
            "total": len(errors),
            "by_type": by_type,
            "recent": list(reversed(errors))[:50],
        }

    def get_security_events(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._security)
        by_type: dict[str, int] = {}
        for e in events:
            by_type[e["event_type"]] = by_type.get(e["event_type"], 0) + 1
        return {
            "total": len(events),
            "by_type": by_type,
            "recent": list(reversed(events))[:50],
        }

    def export_all(self) -> dict[str, Any]:
        return {
            "exported_at": time.time(),
            "system": self.get_system_stats(),
            "requests": self.get_request_analytics(),
            "moderation": self.get_moderation_analytics(),
            "models": self.get_model_stats(),
            "performance": self.get_performance_stats(),
            "errors": self.get_errors(),
            "security": self.get_security_events(),
        }


# Global singleton — import this everywhere
monitor = AegisMonitor()
