"""Tests for moderation visibility helpers."""

from unittest import TestCase

from visibility import (
    HIDDEN_MODERATION_STATUSES,
    VISIBLE_MODERATION_STATUSES,
    filter_visible_posts,
    is_post_visible,
    normalize_moderation_status,
)


class VisibilityTests(TestCase):
    def test_visible_statuses_are_visible(self) -> None:
        for status in VISIBLE_MODERATION_STATUSES:
            with self.subTest(status=status):
                self.assertTrue(is_post_visible(status))
                self.assertTrue(is_post_visible({"moderation_status": status}))

    def test_hidden_statuses_are_not_visible(self) -> None:
        for status in HIDDEN_MODERATION_STATUSES:
            with self.subTest(status=status):
                self.assertFalse(is_post_visible(status))
                self.assertFalse(is_post_visible({"moderation_status": status}))

    def test_unknown_or_missing_status_is_not_visible(self) -> None:
        self.assertFalse(is_post_visible("UNKNOWN"))
        self.assertFalse(is_post_visible(None))
        self.assertFalse(is_post_visible({}))

    def test_status_is_normalized(self) -> None:
        self.assertEqual(
            normalize_moderation_status(" admin_approved "),
            "ADMIN_APPROVED",
        )
        self.assertTrue(is_post_visible(" admin_approved "))

    def test_filter_visible_posts_returns_only_visible_posts(self) -> None:
        posts = [
            {"id": "post-1", "moderation_status": "APPROVED"},
            {"id": "post-2", "moderation_status": "PENDING"},
            {"id": "post-3", "moderation_status": "ADMIN_APPROVED"},
            {"id": "post-4", "moderation_status": "REJECTED"},
        ]

        visible_posts = filter_visible_posts(posts)

        self.assertEqual(
            [post["id"] for post in visible_posts],
            ["post-1", "post-3"],
        )

    def test_filter_visible_posts_normalizes_statuses(self) -> None:
        posts = [
            {"id": "post-1", "moderation_status": " approved "},
            {"id": "post-2", "moderation_status": "under_review"},
        ]

        visible_posts = filter_visible_posts(posts)

        self.assertEqual([post["id"] for post in visible_posts], ["post-1"])

    def test_filter_visible_posts_returns_empty_list_for_empty_input(self) -> None:
        self.assertEqual(filter_visible_posts([]), [])
