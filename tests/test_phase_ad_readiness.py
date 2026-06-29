"""Production-readiness tests for Phase A-D moderation service changes."""

import base64
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
import time
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient
import jwt

import heartbeat_service
import main
import security
from pipeline.safety_flags import ModerationPipelineResult


VALID_POST_ID = "11111111-1111-4111-8111-111111111111"


def _unsigned_jwt(payload):
    header = {"alg": "none", "typ": "JWT"}

    def encode(value):
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode(header)}.{encode(payload)}."


def _admin_headers(role="admin"):
    token = _unsigned_jwt(
        {
            "sub": VALID_POST_ID,
            "app_metadata": {"role": role},
            "exp": int(time.time()) + 3600,
        }
    )
    return {"Authorization": f"Bearer {token}"}


def _signed_admin_headers(secret="test-jwt-secret", role="admin"):
    token = jwt.encode(
        {
            "sub": VALID_POST_ID,
            "app_metadata": {"role": role},
            "exp": int(time.time()) + 3600,
        },
        secret,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


class PhaseAdMigrationTests(TestCase):
    def setUp(self):
        migrations_dir = Path(__file__).resolve().parents[2] / "supabase" / "migrations"
        self.sql = "\n".join(
            [
                (
                    migrations_dir
                    / "202606150003_moderation_phase_a_d_readiness.sql"
                ).read_text(encoding="utf-8"),
                (
                    migrations_dir
                    / "202606150004_moderation_metrics_and_strict_review.sql"
                ).read_text(encoding="utf-8"),
            ]
        )
        self.upper_sql = self.sql.upper()

    def test_approve_rpc_contains_atomic_admin_action_structure(self):
        self.assertIn("CREATE OR REPLACE FUNCTION PUBLIC.APPROVE_REVIEW", self.upper_sql)
        self.assertIn("UPDATE PUBLIC.POSTS", self.upper_sql)
        self.assertIn("SET MODERATION_STATUS = 'ADMIN_APPROVED'", self.upper_sql)
        self.assertIn("INSERT INTO PUBLIC.ADMIN_REVIEW_ACTIONS", self.upper_sql)
        self.assertIn("UPDATE PUBLIC.MODERATION_JOBS", self.upper_sql)
        self.assertIn("STATUS = 'COMPLETED'", self.upper_sql)
        self.assertIn("RAISE EXCEPTION", self.upper_sql)
        self.assertIn("UPDATED_JOB_COUNT", self.upper_sql)
        self.assertIn("NO UNDER_REVIEW MODERATION JOB", self.upper_sql)

    def test_reject_rpc_contains_atomic_admin_action_structure(self):
        self.assertIn("CREATE OR REPLACE FUNCTION PUBLIC.REJECT_REVIEW", self.upper_sql)
        self.assertIn("UPDATE PUBLIC.POSTS", self.upper_sql)
        self.assertIn("SET MODERATION_STATUS = 'ADMIN_REJECTED'", self.upper_sql)
        self.assertIn("INSERT INTO PUBLIC.ADMIN_REVIEW_ACTIONS", self.upper_sql)
        self.assertIn("UPDATE PUBLIC.MODERATION_JOBS", self.upper_sql)
        self.assertIn("STATUS = 'COMPLETED'", self.upper_sql)
        self.assertIn("RAISE EXCEPTION", self.upper_sql)
        self.assertIn("UPDATED_JOB_COUNT", self.upper_sql)
        self.assertIn("NO UNDER_REVIEW MODERATION JOB", self.upper_sql)

    def test_claim_rpc_skips_future_retry_jobs(self):
        self.assertIn("NEXT_ATTEMPT_AT IS NULL OR NEXT_ATTEMPT_AT <= NOW()", self.upper_sql)
        self.assertIn("FOR UPDATE SKIP LOCKED", self.upper_sql)

    def test_audit_metadata_and_worker_heartbeat_schema_exists(self):
        self.assertIn("ADD COLUMN IF NOT EXISTS ADMIN_ID UUID NULL", self.upper_sql)
        self.assertIn("ADD COLUMN IF NOT EXISTS IP_ADDRESS INET NULL", self.upper_sql)
        self.assertIn("ADD COLUMN IF NOT EXISTS REQUEST_ID UUID NULL", self.upper_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS PUBLIC.WORKER_HEARTBEATS", self.upper_sql)

    def test_metrics_rpc_functions_exist(self):
        self.assertIn("CREATE OR REPLACE FUNCTION PUBLIC.GET_QUEUE_METRICS", self.upper_sql)
        self.assertIn("GROUP BY STATUS", self.upper_sql)
        self.assertIn(
            "CREATE OR REPLACE FUNCTION PUBLIC.GET_MODERATION_DECISION_COUNTS_TODAY",
            self.upper_sql,
        )
        self.assertIn("PUBLIC.MODERATION_RESULTS", self.upper_sql)
        self.assertIn("PUBLIC.ADMIN_REVIEW_ACTIONS", self.upper_sql)


class AdminReviewActionsMigrationTests(TestCase):
    """Schema tests for the admin_review_actions table.

    Previously lived in test_admin_review_service.py alongside now-deleted
    dead code (admin_review_service.py).  Moved here because the table is
    used by the approve_review / reject_review RPCs tested in this module.
    """

    def setUp(self):
        migrations_dir = Path(__file__).resolve().parents[2] / "supabase" / "migrations"
        migration_path = migrations_dir / "202606130003_create_admin_review_actions.sql"
        if not migration_path.exists():
            self.skipTest(f"Migration not found: {migration_path}")
        self.sql = migration_path.read_text(encoding="utf-8").upper()

    def test_admin_review_actions_migration_has_required_schema(self):
        self.assertIn("CREATE TABLE IF NOT EXISTS PUBLIC.ADMIN_REVIEW_ACTIONS", self.sql)
        self.assertIn("ID UUID PRIMARY KEY DEFAULT GEN_RANDOM_UUID()", self.sql)
        self.assertIn("POST_ID UUID NOT NULL REFERENCES PUBLIC.POSTS(ID)", self.sql)
        self.assertIn("ADMIN_ACTION TEXT NOT NULL", self.sql)
        self.assertIn("CHECK (ADMIN_ACTION IN ('APPROVE', 'REJECT'))", self.sql)
        self.assertIn("REASON TEXT NULL", self.sql)
        self.assertIn("CREATED_AT TIMESTAMPTZ DEFAULT NOW()", self.sql)
        self.assertIn("IDX_ADMIN_REVIEW_ACTIONS_POST_ID", self.sql)
        self.assertIn("IDX_ADMIN_REVIEW_ACTIONS_CREATED_AT", self.sql)


class CompleteJobAuditColumnsMigrationTests(TestCase):
    """Regression tests for the missing category/ocr audit columns in
    complete_moderation_job.

    Worker-moderated posts store category_name, category_confidence, and
    ocr_text only in posts, not in moderation_results.  The admin review queue
    reads moderation_results for audit purposes and showed zeros for all
    worker-originated jobs.  Migration 202606150007 fixes this by rewriting
    complete_moderation_job to include those columns in the moderation_results
    INSERT.
    """

    def setUp(self):
        migrations_dir = Path(__file__).resolve().parents[2] / "supabase" / "migrations"
        migration_path = migrations_dir / "202606150007_fix_complete_job_audit_columns.sql"
        self.sql = migration_path.read_text(encoding="utf-8")
        self.upper_sql = self.sql.upper()

    def test_fix_migration_rewrites_complete_moderation_job(self):
        self.assertIn("CREATE OR REPLACE FUNCTION PUBLIC.COMPLETE_MODERATION_JOB", self.upper_sql)

    def test_moderation_results_insert_includes_category_name(self):
        self.assertIn("CATEGORY_NAME", self.upper_sql)
        # Must appear in VALUES clause context, not just in parameters
        self.assertIn("P_CATEGORY_NAME", self.upper_sql)

    def test_moderation_results_insert_includes_category_confidence(self):
        self.assertIn("CATEGORY_CONFIDENCE", self.upper_sql)
        self.assertIn("P_CATEGORY_CONFIDENCE", self.upper_sql)

    def test_moderation_results_insert_includes_ocr_text(self):
        self.assertIn("OCR_TEXT", self.upper_sql)
        self.assertIn("P_OCR_TEXT", self.upper_sql)

    def test_function_is_security_definer_and_service_role_only(self):
        self.assertIn("SECURITY DEFINER", self.upper_sql)
        self.assertIn("GRANT EXECUTE", self.upper_sql)
        self.assertIn("TO SERVICE_ROLE", self.upper_sql)
        self.assertIn("REVOKE ALL", self.upper_sql)

    def test_complete_job_still_updates_posts_with_category_and_ocr(self):
        self.assertIn("UPDATE PUBLIC.POSTS", self.upper_sql)
        self.assertIn("CATEGORY_NAME = P_CATEGORY_NAME", self.upper_sql)
        self.assertIn("CATEGORY_CONFIDENCE = P_CATEGORY_CONFIDENCE", self.upper_sql)
        self.assertIn("OCR_TEXT = P_OCR_TEXT", self.upper_sql)


class HeartbeatTests(TestCase):
    def test_worker_heartbeat_upserts_status(self):
        client = _HeartbeatClient([])

        heartbeat_service.record_worker_heartbeat(
            "worker-1",
            jobs_processed=7,
            gpu_id="0",
            client=client,
        )

        self.assertEqual(client.rows[0]["worker_id"], "worker-1")
        self.assertEqual(client.rows[0]["jobs_processed"], 7)
        self.assertEqual(client.rows[0]["gpu_id"], "0")
        self.assertEqual(client.queries[0].upsert_conflict, "worker_id")

    def test_worker_status_online_threshold(self):
        now = datetime.now(UTC)
        client = _HeartbeatClient(
            [
                {
                    "worker_id": "worker-1",
                    "last_seen": (now - timedelta(seconds=10)).isoformat(),
                    "jobs_processed": 5,
                },
                {
                    "worker_id": "worker-2",
                    "last_seen": (now - timedelta(seconds=120)).isoformat(),
                    "jobs_processed": 2,
                },
            ]
        )

        statuses = heartbeat_service.get_worker_statuses(client=client)

        self.assertTrue(statuses[0]["online"])
        self.assertFalse(statuses[1]["online"])


class SecurityAndDeploymentTests(TestCase):
    def setUp(self):
        os.environ["API_SHARED_SECRET"] = "test-secret"
        os.environ.pop("SUPABASE_JWT_SECRET", None)
        os.environ["ALLOW_INSECURE_ADMIN_AUTH"] = "true"
        security.reset_rate_limits_for_tests()
        self.client = TestClient(main.app)

    def test_secret_missing_and_insecure_false_is_denied(self):
        os.environ.pop("SUPABASE_JWT_SECRET", None)
        os.environ["ALLOW_INSECURE_ADMIN_AUTH"] = "false"

        response = self.client.get(
            "/admin/moderation/metrics",
            headers=_admin_headers(),
        )

        self.assertEqual(response.status_code, 500)
        self.assertIn("Admin authentication is not configured", response.json()["detail"])

    def test_secret_missing_and_insecure_true_is_allowed(self):
        os.environ.pop("SUPABASE_JWT_SECRET", None)
        os.environ["ALLOW_INSECURE_ADMIN_AUTH"] = "true"

        with patch(
            "main.queue_service.get_queue_depth_by_status",
            return_value={"PENDING": 0, "PROCESSING": 0, "UNDER_REVIEW": 0, "FAILED": 0},
        ), patch(
            "main.queue_service.get_decision_counts_today",
            return_value={"approved_today": 0, "rejected_today": 0},
        ):
            response = self.client.get(
                "/admin/moderation/metrics",
                headers=_admin_headers(),
            )

        self.assertEqual(response.status_code, 200)

    def test_invalid_admin_jwt_is_rejected(self):
        os.environ["SUPABASE_JWT_SECRET"] = "test-jwt-secret"
        os.environ["ALLOW_INSECURE_ADMIN_AUTH"] = "false"

        response = self.client.get(
            "/admin/moderation/metrics",
            headers={"Authorization": "Bearer not-a-jwt"},
        )

        self.assertEqual(response.status_code, 401)

    def test_admin_jwt_is_accepted(self):
        os.environ["SUPABASE_JWT_SECRET"] = "test-jwt-secret"
        os.environ["ALLOW_INSECURE_ADMIN_AUTH"] = "false"

        with patch(
            "main.queue_service.get_queue_depth_by_status",
            return_value={"PENDING": 0, "PROCESSING": 0, "UNDER_REVIEW": 0, "FAILED": 0},
        ), patch(
            "main.queue_service.get_decision_counts_today",
            return_value={"approved_today": 0, "rejected_today": 0},
        ):
            response = self.client.get(
                "/admin/moderation/metrics",
                headers=_signed_admin_headers(),
            )

        self.assertEqual(response.status_code, 200)

    def test_invalid_admin_requests_do_not_consume_valid_admin_bucket(self):
        os.environ["SUPABASE_JWT_SECRET"] = "test-jwt-secret"
        os.environ["ALLOW_INSECURE_ADMIN_AUTH"] = "false"

        for _ in range(30):
            self.client.get(
                "/admin/moderation/metrics",
                headers={"Authorization": "Bearer invalid"},
            )

        with patch(
            "main.queue_service.get_queue_depth_by_status",
            return_value={"PENDING": 0, "PROCESSING": 0, "UNDER_REVIEW": 0, "FAILED": 0},
        ), patch(
            "main.queue_service.get_decision_counts_today",
            return_value={"approved_today": 0, "rejected_today": 0},
        ):
            response = self.client.get(
                "/admin/moderation/metrics",
                headers=_signed_admin_headers(),
            )

        self.assertEqual(response.status_code, 200)

    def test_moderate_endpoint_rate_limit(self):
        result = ModerationPipelineResult(
            scores={
                "adult_score": 0.0,
                "heritage_score": 0.0,
                "child_safety_score": 0.0,
                "violence_self_harm_score": 0.0,
                "content_quality_score": 0.0,
            },
            category_scores={},
            ocr_text="",
        )

        with patch("main._download_image", return_value="tmp.jpg"), patch(
            "main._validate_image_file"
        ), patch("main.Path.unlink"), patch(
            "main.analyze_image",
            return_value=result,
        ), patch("main.write_results"):
            responses = [
                self.client.post(
                    "/moderate",
                    headers={"X-API-Key": "test-secret"},
                    json={
                        "post_id": VALID_POST_ID,
                        "image_url": "https://example.com/a.jpg",
                        "caption": "caption",
                    },
                )
                for _ in range(61)
            ]

        self.assertEqual(responses[59].status_code, 200)
        self.assertEqual(responses[60].status_code, 429)

    def test_admin_endpoint_rate_limit(self):
        with patch(
            "main.queue_service.get_queue_depth_by_status",
            return_value={"PENDING": 0, "PROCESSING": 0, "UNDER_REVIEW": 0, "FAILED": 0},
        ), patch(
            "main.queue_service.get_decision_counts_today",
            return_value={"approved_today": 0, "rejected_today": 0},
        ):
            responses = [
                self.client.get(
                    "/admin/moderation/metrics",
                    headers=_admin_headers(),
                )
                for _ in range(31)
            ]

        self.assertEqual(responses[29].status_code, 200)
        self.assertEqual(responses[30].status_code, 429)

    def test_dockerfile_and_compose_define_api_and_worker(self):
        service_root = Path(__file__).resolve().parents[1]
        dockerfile = (service_root / "Dockerfile").read_text(encoding="utf-8")
        compose = (service_root / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn("uvicorn", dockerfile)
        self.assertIn("PRELOAD_MODELS", dockerfile)
        self.assertIn("model_warmup.warmup_models()", dockerfile)
        self.assertIn("moderation-api", compose)
        self.assertIn("moderation-worker", compose)
        self.assertIn("python worker.py", compose)


class _HeartbeatClient:
    def __init__(self, rows):
        self.rows = list(rows)
        self.queries = []

    def table(self, table_name):
        query = _HeartbeatQuery(self, table_name)
        self.queries.append(query)
        return query


class _HeartbeatQuery:
    def __init__(self, client, table_name):
        self.client = client
        self.table_name = table_name
        self.upsert_payload = None
        self.upsert_conflict = None
        self.select_columns = None

    def upsert(self, payload, on_conflict=None):
        self.upsert_payload = payload
        self.upsert_conflict = on_conflict
        return self

    def select(self, columns):
        self.select_columns = columns
        return self

    def execute(self):
        if self.upsert_payload is not None:
            existing = [
                row
                for row in self.client.rows
                if row.get("worker_id") == self.upsert_payload.get("worker_id")
            ]
            if existing:
                existing[0].update(self.upsert_payload)
            else:
                self.client.rows.append(dict(self.upsert_payload))
            return _Response([self.upsert_payload])
        return _Response([dict(row) for row in self.client.rows])


class _Response:
    def __init__(self, data):
        self.data = data
