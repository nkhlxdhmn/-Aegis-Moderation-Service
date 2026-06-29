"""Helpers for deciding whether moderated posts may be shown."""

from __future__ import annotations

from typing import Any, Mapping

VISIBLE_MODERATION_STATUSES = frozenset({"APPROVED", "ADMIN_APPROVED"})
HIDDEN_MODERATION_STATUSES = frozenset(
    {
        "PENDING",
        "PROCESSING",
        "UNDER_REVIEW",
        "REJECTED",
        "ADMIN_REJECTED",
    }
)


def normalize_moderation_status(status: Any) -> str:
    """Return a normalized moderation status string."""

    return str(status or "").strip().upper()


def is_post_visible(post_or_status: Mapping[str, Any] | str | None) -> bool:
    """Return True only for moderation statuses that may be visible."""

    if isinstance(post_or_status, Mapping):
        status = post_or_status.get("moderation_status")
    else:
        status = post_or_status

    return normalize_moderation_status(status) in VISIBLE_MODERATION_STATUSES


def filter_visible_posts(
    posts: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return only posts with moderation statuses that may be visible."""

    return [post for post in posts if is_post_visible(post)]
