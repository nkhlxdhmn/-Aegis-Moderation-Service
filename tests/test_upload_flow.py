"""Tests for image post upload moderation queue integration."""

from pathlib import Path
from unittest import TestCase


REPO_ROOT = Path(__file__).resolve().parents[2]


class UploadFlowTests(TestCase):
    def test_image_post_trigger_sets_pending_and_enqueues_job(self):
        migration = (
            REPO_ROOT
            / "supabase"
            / "migrations"
            / "202606150002_enqueue_image_post_moderation_jobs.sql"
        )

        sql = migration.read_text(encoding="utf-8").upper()

        self.assertIn("CREATE OR REPLACE FUNCTION PUBLIC.SET_IMAGE_POST_PENDING_MODERATION", sql)
        self.assertIn("IF NEW.POST_TYPE = 'IMAGE' THEN", sql)
        self.assertIn("NEW.MODERATION_STATUS = 'PENDING'", sql)
        self.assertIn("BEFORE INSERT ON PUBLIC.POSTS", sql)
        self.assertIn("CREATE OR REPLACE FUNCTION PUBLIC.ENQUEUE_IMAGE_POST_MODERATION_JOB", sql)
        self.assertIn("INSERT INTO PUBLIC.MODERATION_JOBS", sql)
        self.assertIn("NEW.MEDIA_URLS[1]", sql)
        self.assertIn("'PENDING'", sql)
        self.assertIn("AFTER INSERT ON PUBLIC.POSTS", sql)

    def test_moderation_status_column_supports_queue_and_admin_states(self):
        migration = (
            REPO_ROOT
            / "supabase"
            / "migrations"
            / "202606150002_enqueue_image_post_moderation_jobs.sql"
        )

        sql = migration.read_text(encoding="utf-8").upper()

        self.assertIn("ADD COLUMN IF NOT EXISTS MODERATION_STATUS TEXT DEFAULT 'APPROVED'", sql)
        for status in [
            "PENDING",
            "PROCESSING",
            "UNDER_REVIEW",
            "APPROVED",
            "REJECTED",
            "ADMIN_APPROVED",
            "ADMIN_REJECTED",
        ]:
            self.assertIn(f"'{status}'", sql)

    def test_flutter_create_post_does_not_run_synchronous_moderation(self):
        post_service = REPO_ROOT / "lib" / "services" / "post_service.dart"

        if not post_service.exists():
            self.skipTest(
                "Flutter source (lib/services/post_service.dart) is not present in "
                "this repository checkout; skipping Flutter/backend contract check."
            )

        source = post_service.read_text(encoding="utf-8")
        create_post_source = source[
            source.index("Future<Map<String, dynamic>> createPost")
            : source.index("  /// Creates a repost for an existing post.")
        ]

        self.assertIn(".from('posts')", create_post_source)
        self.assertIn(".insert(postData)", create_post_source)
        self.assertNotIn("/moderate", create_post_source)
        self.assertNotIn("analyze_image", create_post_source)
        self.assertNotIn("moderation_results", create_post_source)
