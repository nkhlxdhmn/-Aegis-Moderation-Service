"""Authentication, request context, and rate limiting for moderation APIs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import base64
import hmac
import json
import os
import time
from typing import Any
from uuid import uuid4

from fastapi import Header, HTTPException, Request, status


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    admin_id: str | None = None


class RateLimitError(RuntimeError):
    """Raised when a request exceeds its configured rate limit."""


class InMemoryRateLimiter:
    """Small fixed-window limiter used for tests and local development."""

    def __init__(self) -> None:
        self._windows: dict[str, tuple[float, int]] = {}

    def check(self, key: str, limit: int, window_seconds: int = 60) -> None:
        now = time.time()
        window_start, count = self._windows.get(key, (now, 0))
        if now - window_start >= window_seconds:
            window_start, count = now, 0
        count += 1
        self._windows[key] = (window_start, count)
        if count > limit:
            raise RateLimitError("Rate limit exceeded.")


_memory_limiter = InMemoryRateLimiter()
ADMIN_RATE_LIMIT = 30


def request_id() -> str:
    return str(uuid4())


def client_ip(request: Request) -> str | None:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else None


def check_rate_limit(request: Request, bucket: str, limit: int) -> None:
    identifier = client_ip(request) or "unknown"
    check_rate_limit_key(f"{bucket}:{identifier}", limit)


def check_rate_limit_key(key: str, limit: int) -> None:
    try:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            _check_redis_rate_limit(redis_url, key, limit)
        else:
            _memory_limiter.check(key, limit)
    except RateLimitError:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
        )


_redis_pool: Any | None = None
_redis_pool_url: str | None = None


def _get_redis_pool(redis_url: str) -> Any:
    """Return a process-wide Redis connection pool, creating it once per URL."""
    global _redis_pool, _redis_pool_url
    if _redis_pool is None or _redis_pool_url != redis_url:
        import redis as _redis
        _redis_pool = _redis.ConnectionPool.from_url(redis_url, max_connections=20)
        _redis_pool_url = redis_url
    return _redis_pool


def _check_redis_rate_limit(redis_url: str, key: str, limit: int) -> None:
    try:
        import redis
    except ImportError:
        _memory_limiter.check(key, limit)
        return

    client = redis.Redis(connection_pool=_get_redis_pool(redis_url))
    redis_key = f"moderation-rate:{key}:{int(time.time() // 60)}"
    count = client.incr(redis_key)
    if count == 1:
        client.expire(redis_key, 60)
    if int(count) > limit:
        raise RateLimitError("Rate limit exceeded.")


def require_api_key(x_api_key: str | None) -> None:
    shared_secret = os.getenv("API_SHARED_SECRET")
    if not shared_secret or not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    if not hmac.compare_digest(x_api_key, shared_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )


def require_admin(
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> RequestContext:
    try:
        payload = _decode_admin_jwt(authorization)
    except HTTPException as exc:
        if exc.status_code in {
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        }:
            identifier = client_ip(request) or "unknown"
            check_rate_limit_key(f"admin-unauth:{identifier}", ADMIN_RATE_LIMIT)
        raise

    role = (
        (payload.get("app_metadata") or {}).get("role")
        or (payload.get("user_metadata") or {}).get("role")
        or payload.get("role")
    )
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role is required.",
        )
    admin_id = str(payload.get("sub") or "")
    check_rate_limit_key(f"admin:{admin_id}", ADMIN_RATE_LIMIT)
    return RequestContext(request_id=request_id(), admin_id=admin_id)


def _decode_admin_jwt(authorization: str | None) -> dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin bearer token is required.",
        )
    token = authorization.split(" ", 1)[1].strip()
    jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
    if jwt_secret:
        try:
            import jwt

            return dict(
                jwt.decode(
                    token,
                    jwt_secret,
                    algorithms=["HS256"],
                    options={"verify_aud": False},
                )
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid admin token.",
            ) from exc
    if os.getenv("ALLOW_INSECURE_ADMIN_AUTH", "false").lower() == "true":
        if os.getenv("ENVIRONMENT", "local").lower() not in {
            "local",
            "development",
            "test",
        }:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Unsigned admin auth is disabled outside local/test environments.",
            )
        return _decode_unsigned_test_jwt(token)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=(
            "Admin authentication is not configured. Set SUPABASE_JWT_SECRET "
            "or ALLOW_INSECURE_ADMIN_AUTH=true for local tests."
        ),
    )


def _decode_unsigned_test_jwt(token: str) -> dict[str, Any]:
    """Decode unsigned JWT payload only when no JWT secret is configured."""

    try:
        parts = token.split(".")
        if len(parts) < 2:
            raise ValueError("JWT must have at least two parts")
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded.decode("utf-8"))
        exp = data.get("exp")
        if exp is not None and datetime.fromtimestamp(float(exp), UTC) < datetime.now(UTC):
            raise ValueError("JWT expired")
        return dict(data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token.",
        ) from exc


def reset_rate_limits_for_tests() -> None:
    _memory_limiter._windows.clear()
