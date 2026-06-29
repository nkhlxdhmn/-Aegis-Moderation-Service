"""Endpoint-level tests for moderation activation and reason codes."""

import base64
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

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


class ModerateEndpointTests(TestCase):
    def setUp(self) -> None:
        os.environ["API_SHARED_SECRET"] = "test-secret"
        os.environ.pop("SUPABASE_JWT_SECRET", None)
        os.environ["ALLOW_INSECURE_ADMIN_AUTH"] = "true"
        security.reset_rate_limits_for_tests()
        self.client = TestClient(main.app)

    def _admin_headers(self, role="admin"):
        payload = {
            "sub": VALID_POST_ID,
            "app_metadata": {"role": role},
            "exp": int(time.time()) + 3600,
        }
        token = _unsigned_jwt(payload)
        return {"Authorization": f"Bearer {token}"}

    def test_pipeline_error_forces_under_review(self) -> None:
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
            pipeline_error=True,
            error_reason="NudeNet failed",
        )

        with patch("main._download_image", return_value="tmp.jpg"), patch(
            "main._validate_image_file"
        ), patch("main.Path.unlink"), patch("main.analyze_image", return_value=result), patch(
            "main.write_results"
        ):
            response = self.client.post(
                "/moderate",
                headers={"X-API-Key": "test-secret"},
                json={
                    "post_id": VALID_POST_ID,
                    "image_url": "https://example.com/a.jpg",
                    "caption": "caption",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["decision"], "UNDER_REVIEW")
        self.assertTrue(body["reason"].startswith("PIPELINE_ERROR:"))

    def test_heritage_exception_reason_code(self) -> None:
        result = ModerationPipelineResult(
            scores={
                "adult_score": 0.85,
                "heritage_score": 0.9,
                "child_safety_score": 0.0,
                "violence_self_harm_score": 0.0,
                "content_quality_score": 0.0,
            },
            category_scores={"Religious & Spiritual Heritage": 0.9},
            ocr_text="",
        )

        with patch("main._download_image", return_value="tmp.jpg"), patch(
            "main._validate_image_file"
        ), patch("main.Path.unlink"), patch("main.analyze_image", return_value=result), patch(
            "main.write_results"
        ):
            response = self.client.post(
                "/moderate",
                headers={"X-API-Key": "test-secret"},
                json={
                    "post_id": VALID_POST_ID,
                    "image_url": "https://example.com/a.jpg",
                    "caption": "Khajuraho sculpture",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["decision"], "UNDER_REVIEW")
        self.assertTrue(body["reason"].startswith("HERITAGE_REVIEW:"))

    def test_nsfw_rejection_reason_code(self) -> None:
        result = ModerationPipelineResult(
            scores={
                "adult_score": 0.85,
                "heritage_score": 0.1,
                "child_safety_score": 0.0,
                "violence_self_harm_score": 0.0,
                "content_quality_score": 0.0,
            },
            category_scores={},
            ocr_text="",
        )

        with patch("main._download_image", return_value="tmp.jpg"), patch(
            "main._validate_image_file"
        ), patch("main.Path.unlink"), patch("main.analyze_image", return_value=result), patch(
            "main.write_results"
        ):
            response = self.client.post(
                "/moderate",
                headers={"X-API-Key": "test-secret"},
                json={
                    "post_id": VALID_POST_ID,
                    "image_url": "https://example.com/a.jpg",
                    "caption": "caption",
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["decision"], "REJECTED")
        self.assertTrue(body["reason"].startswith("NSFW_CONTENT:"))

    def test_invalid_post_id_returns_422(self) -> None:
        response = self.client.post(
            "/moderate",
            headers={"X-API-Key": "test-secret"},
            json={
                "post_id": "post-1",
                "image_url": "https://example.com/a.jpg",
                "caption": "caption",
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_missing_api_key_returns_401(self) -> None:
        response = self.client.post(
            "/moderate",
            json={
                "post_id": VALID_POST_ID,
                "image_url": "https://example.com/a.jpg",
                "caption": "caption",
            },
        )

        self.assertEqual(response.status_code, 401)

    def test_large_image_is_rejected_before_inference(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file.write(b"0" * (main.MAX_IMAGE_SIZE_BYTES + 1))
            path = file.name

        try:
            with self.assertRaises(main.ImageInputError):
                main._validate_image_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_oversized_dimensions_are_rejected_before_inference(self) -> None:
        from PIL import Image

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as file:
            path = file.name

        try:
            Image.new("RGB", (main.MAX_IMAGE_SIDE_PIXELS + 1, 1)).save(path)
            with self.assertRaises(main.ImageInputError):
                main._validate_image_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_decompression_bomb_warning_is_rejected_before_inference(self) -> None:
        from PIL import Image

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as file:
            path = file.name

        try:
            Image.new("RGB", (2, 2)).save(path)
            with patch("main.MAX_IMAGE_PIXELS", 1):
                with self.assertRaises(main.ImageInputError):
                    main._validate_image_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_http_image_url_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"MODERATION_ALLOWED_IMAGE_HOSTS": "example.com"},
            clear=False,
        ):
            with self.assertRaises(main.ImageInputError):
                main._validate_image_url("http://example.com/a.jpg")

    def test_private_image_url_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"MODERATION_ALLOWED_IMAGE_HOSTS": "localhost"},
            clear=False,
        ), patch(
            "main.socket.getaddrinfo",
            return_value=[
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("127.0.0.1", 0),
                )
            ],
        ):
            with self.assertRaises(main.ImageInputError):
                main._validate_image_url("https://localhost/a.jpg")

    def test_link_local_image_url_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"MODERATION_ALLOWED_IMAGE_HOSTS": "metadata.local"},
            clear=False,
        ), patch(
            "main.socket.getaddrinfo",
            return_value=[
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("169.254.169.254", 0),
                )
            ],
        ):
            with self.assertRaises(main.ImageInputError):
                main._validate_image_url("https://metadata.local/a.jpg")

    def test_unlisted_image_host_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"MODERATION_ALLOWED_IMAGE_HOSTS": "media.example.com"},
            clear=False,
        ):
            with self.assertRaises(main.ImageInputError):
                main._validate_image_url("https://example.com/a.jpg")

    def test_supabase_host_is_allowed_by_default(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SUPABASE_URL": "https://project.supabase.co",
                "MODERATION_ALLOWED_IMAGE_HOSTS": "",
            },
            clear=False,
        ), patch(
            "main.socket.getaddrinfo",
            return_value=[
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("93.184.216.34", 0),
                )
            ],
        ):
            main._validate_image_url("https://project.supabase.co/storage/a.jpg")

    def test_unsafe_redirect_target_is_rejected(self) -> None:
        handler = main._ValidatingRedirectHandler()
        request = main.Request("https://project.supabase.co/storage/a.jpg")

        with patch.dict(
            os.environ,
            {"MODERATION_ALLOWED_IMAGE_HOSTS": "project.supabase.co"},
            clear=False,
        ):
            with self.assertRaises(main.ImageInputError):
                handler.redirect_request(
                    request,
                    None,
                    302,
                    "Found",
                    {},
                    "https://127.0.0.1/private.jpg",
                )

    def test_corrupted_image_is_rejected_before_inference(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file.write(b"not an image")
            path = file.name

        try:
            with self.assertRaises(main.ImageInputError):
                main._validate_image_file(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_admin_review_queue_returns_under_review_jobs_with_scores(self) -> None:
        client = _FakeSupabaseClient(
            {
                "moderation_jobs": [
                    {
                        "post_id": VALID_POST_ID,
                        "image_url": "https://project.supabase.co/storage/a.jpg",
                        "status": "UNDER_REVIEW",
                        "created_at": "2026-06-13T00:00:00+00:00",
                    },
                    {
                        "post_id": "22222222-2222-4222-8222-222222222222",
                        "image_url": "https://project.supabase.co/storage/b.jpg",
                        "status": "COMPLETED",
                        "created_at": "2026-06-13T00:01:00+00:00",
                    },
                ],
                "moderation_results": [
                    {
                        "post_id": VALID_POST_ID,
                        "adult_score": 0.1,
                        "heritage_score": 0.8,
                        "child_safety_score": 0.0,
                        "violence_self_harm_score": 0.0,
                        "content_quality_score": 0.3,
                        "decision": "UNDER_REVIEW",
                        "reason": "HERITAGE_REVIEW: Review needed.",
                        "created_at": "2026-06-13T00:02:00+00:00",
                    }
                ],
            }
        )

        with patch("main.get_supabase_client", return_value=client):
            response = self.client.get(
                "/admin/review-queue?limit=10&offset=0",
                headers=self._admin_headers(),
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["limit"], 10)
        self.assertEqual(body["offset"], 0)
        self.assertEqual(body["items"][0]["post_id"], VALID_POST_ID)
        self.assertEqual(body["items"][0]["decision"], "UNDER_REVIEW")
        self.assertEqual(body["items"][0]["adult_score"], 0.1)
        self.assertEqual(body["items"][0]["child_safety_score"], 0.0)
        self.assertEqual(body["items"][0]["violence_self_harm_score"], 0.0)
        self.assertEqual(body["items"][0]["heritage_score"], 0.8)
        self.assertEqual(body["items"][0]["scores"]["heritage_score"], 0.8)

        jobs_query = client.operations[0]
        self.assertEqual(jobs_query.table_name, "moderation_jobs")
        self.assertIn(("eq", "status", "UNDER_REVIEW"), jobs_query.filters)
        self.assertEqual(jobs_query.range_args, (0, 9))

    def test_admin_review_detail_returns_post_result_ocr_and_image(self) -> None:
        client = _FakeSupabaseClient(
            {
                "posts": [
                    {
                        "id": VALID_POST_ID,
                        "caption": "caption",
                        "ocr_text": "temple inscription",
                        "moderation_status": "UNDER_REVIEW",
                    }
                ],
                "moderation_results": [
                    {
                        "post_id": VALID_POST_ID,
                        "decision": "UNDER_REVIEW",
                        "reason": "PII_DETECTED: Review needed.",
                        "adult_score": 0.0,
                        "heritage_score": 0.2,
                        "created_at": "2026-06-13T00:02:00+00:00",
                    }
                ],
                "moderation_jobs": [
                    {
                        "post_id": VALID_POST_ID,
                        "image_url": "https://project.supabase.co/storage/a.jpg",
                        "created_at": "2026-06-13T00:00:00+00:00",
                    }
                ],
            }
        )

        with patch("main.get_supabase_client", return_value=client):
            response = self.client.get(
                f"/admin/review/{VALID_POST_ID}",
                headers=self._admin_headers(),
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["post"]["id"], VALID_POST_ID)
        self.assertEqual(body["moderation_result"]["decision"], "UNDER_REVIEW")
        self.assertEqual(body["ocr_text"], "temple inscription")
        self.assertEqual(body["image_url"], "https://project.supabase.co/storage/a.jpg")
        self.assertEqual(body["decision"], "UNDER_REVIEW")
        self.assertEqual(body["reason"], "PII_DETECTED: Review needed.")

    def test_admin_review_detail_returns_404_when_post_missing(self) -> None:
        client = _FakeSupabaseClient(
            {
                "posts": [],
                "moderation_results": [],
                "moderation_jobs": [],
            }
        )

        with patch("main.get_supabase_client", return_value=client):
            response = self.client.get(
                f"/admin/review/{VALID_POST_ID}",
                headers=self._admin_headers(),
            )

        self.assertEqual(response.status_code, 404)

    def test_admin_review_queue_requires_admin_token(self) -> None:
        response = self.client.get("/admin/review-queue")

        self.assertEqual(response.status_code, 401)

    def test_admin_review_queue_rejects_non_admin_token(self) -> None:
        response = self.client.get(
            "/admin/review-queue",
            headers=self._admin_headers(role="user"),
        )

        self.assertEqual(response.status_code, 403)

    def test_admin_approve_updates_post_inserts_action_and_completes_job(self) -> None:
        client = _FakeSupabaseClient(
            {
                "posts": [{"id": VALID_POST_ID, "moderation_status": "UNDER_REVIEW"}],
                "admin_review_actions": [],
                "moderation_jobs": [
                    {"post_id": VALID_POST_ID, "status": "UNDER_REVIEW"}
                ],
            }
        )

        with patch("main.get_supabase_client", return_value=client):
            response = self.client.post(
                f"/admin/review/{VALID_POST_ID}/approve",
                headers=self._admin_headers(),
                json={"reason": "Looks acceptable."},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            client.rows["posts"][0]["moderation_status"],
            "ADMIN_APPROVED",
        )
        self.assertEqual(
            client.rows["admin_review_actions"][0]["admin_action"],
            "APPROVE",
        )
        self.assertEqual(
            client.rows["admin_review_actions"][0]["reason"],
            "Looks acceptable.",
        )
        self.assertEqual(client.rows["moderation_jobs"][0]["status"], "COMPLETED")
        self.assertEqual(response.json()["job_status"], "COMPLETED")
        self.assertEqual(response.json()["admin_action"], "APPROVE")
        self.assertEqual(client.rpc_calls[0].function_name, "approve_review")
        self.assertEqual(client.rpc_calls[0].params["p_post_id"], VALID_POST_ID)
        self.assertEqual(
            client.rpc_calls[0].params["p_admin_id"],
            VALID_POST_ID,
        )
        self.assertIsNotNone(client.rpc_calls[0].params["p_request_id"])

    def test_admin_reject_updates_post_inserts_action_and_completes_job(self) -> None:
        client = _FakeSupabaseClient(
            {
                "posts": [{"id": VALID_POST_ID, "moderation_status": "UNDER_REVIEW"}],
                "admin_review_actions": [],
                "moderation_jobs": [
                    {"post_id": VALID_POST_ID, "status": "UNDER_REVIEW"}
                ],
            }
        )

        with patch("main.get_supabase_client", return_value=client):
            response = self.client.post(
                f"/admin/review/{VALID_POST_ID}/reject",
                headers=self._admin_headers(),
                json={"reason": "Policy violation."},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            client.rows["posts"][0]["moderation_status"],
            "ADMIN_REJECTED",
        )
        self.assertEqual(
            client.rows["admin_review_actions"][0]["admin_action"],
            "REJECT",
        )
        self.assertEqual(
            client.rows["admin_review_actions"][0]["reason"],
            "Policy violation.",
        )
        self.assertEqual(client.rows["moderation_jobs"][0]["status"], "COMPLETED")
        self.assertEqual(response.json()["admin_action"], "REJECT")
        self.assertEqual(client.rpc_calls[0].function_name, "reject_review")

    def test_admin_approve_missing_post_returns_404(self) -> None:
        client = _FakeSupabaseClient(
            {
                "posts": [],
                "admin_review_actions": [],
                "moderation_jobs": [
                    {"post_id": VALID_POST_ID, "status": "UNDER_REVIEW"}
                ],
            }
        )

        with patch("main.get_supabase_client", return_value=client):
            response = self.client.post(
                f"/admin/review/{VALID_POST_ID}/approve",
                headers=self._admin_headers(),
                json={"reason": "Looks acceptable."},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(client.rows["admin_review_actions"], [])
        self.assertEqual(client.rows["moderation_jobs"][0]["status"], "UNDER_REVIEW")
        self.assertEqual(client.rpc_calls[0].function_name, "approve_review")

    def test_admin_rpc_failure_does_not_perform_direct_table_writes(self) -> None:
        client = _FakeSupabaseClient(
            {
                "posts": [{"id": VALID_POST_ID, "moderation_status": "UNDER_REVIEW"}],
                "admin_review_actions": [],
                "moderation_jobs": [
                    {"post_id": VALID_POST_ID, "status": "UNDER_REVIEW"}
                ],
            }
        )
        client.rpc_error = RuntimeError("database exploded")

        with patch("main.get_supabase_client", return_value=client):
            response = self.client.post(
                f"/admin/review/{VALID_POST_ID}/reject",
                headers=self._admin_headers(),
                json={"reason": "Policy violation."},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(client.rows["posts"][0]["moderation_status"], "UNDER_REVIEW")
        self.assertEqual(client.rows["admin_review_actions"], [])
        self.assertEqual(client.rows["moderation_jobs"][0]["status"], "UNDER_REVIEW")
        self.assertEqual(client.operations, [])

    def test_moderation_metrics_returns_expected_counts(self) -> None:
        with patch(
            "main.queue_service.get_queue_depth_by_status",
            return_value={
                "PENDING": 15,
                "PROCESSING": 2,
                "UNDER_REVIEW": 3,
                "FAILED": 1,
                "COMPLETED": 9,
            },
        ), patch(
            "main.queue_service.get_decision_counts_today",
            return_value={"approved_today": 125, "rejected_today": 17},
        ):
            response = self.client.get(
                "/admin/moderation/metrics",
                headers=self._admin_headers(),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "pending": 15,
                "processing": 2,
                "under_review": 3,
                "failed": 1,
                "approved_today": 125,
                "rejected_today": 17,
            },
        )

    def test_moderation_metrics_handles_db_errors(self) -> None:
        with patch(
            "main.queue_service.get_queue_depth_by_status",
            side_effect=RuntimeError("db down"),
        ):
            response = self.client.get(
                "/admin/moderation/metrics",
                headers=self._admin_headers(),
            )

        self.assertEqual(response.status_code, 502)

    def test_health_reports_db_and_model_status(self) -> None:
        client = _FakeSupabaseClient({"moderation_jobs": []})

        with patch("main.get_supabase_client", return_value=client), patch(
            "main.model_warmup.model_status",
            return_value="loaded",
        ):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "healthy",
                "queue": "ok",
                "db": "ok",
                "models": "loaded",
            },
        )

    def test_admin_approve_already_reviewed_job_returns_404(self) -> None:
        # The approve_review RPC raises when no UNDER_REVIEW job exists (already
        # processed).  This must return 404 "no pending review job" so admins
        # know the post is no longer actionable rather than getting a misleading
        # 502.  The old code mapped this to 502 which confused admins into
        # retrying.
        client = _FakeSupabaseClient(
            {
                "posts": [{"id": VALID_POST_ID, "moderation_status": "UNDER_REVIEW"}],
                "admin_review_actions": [],
                "moderation_jobs": [
                    {"post_id": VALID_POST_ID, "status": "COMPLETED"}
                ],
            }
        )

        with patch("main.get_supabase_client", return_value=client):
            response = self.client.post(
                f"/admin/review/{VALID_POST_ID}/approve",
                headers=self._admin_headers(),
                json={"reason": "Looks acceptable."},
            )

        self.assertEqual(response.status_code, 404)
        self.assertIn("pending review", response.json()["detail"].lower())
        # DB must be untouched (RPC transaction rolls back on exception)
        self.assertEqual(client.rows["posts"][0]["moderation_status"], "UNDER_REVIEW")
        self.assertEqual(client.rows["admin_review_actions"], [])
        self.assertEqual(client.rows["moderation_jobs"][0]["status"], "COMPLETED")

    def test_admin_rpc_transient_error_is_retried_before_502(self) -> None:
        # The approve_review RPC call must be retried on transient Supabase
        # failures.  Previously it was a bare .execute() with no retry wrapper.
        client = _FakeSupabaseClient(
            {
                "posts": [{"id": VALID_POST_ID, "moderation_status": "UNDER_REVIEW"}],
                "admin_review_actions": [],
                "moderation_jobs": [
                    {"post_id": VALID_POST_ID, "status": "UNDER_REVIEW"}
                ],
            }
        )
        call_count = {"n": 0}
        original_rpc = client.rpc

        def _flaky_rpc(function_name, params):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: simulate transient 503
                class _FakeFlaky:
                    def execute(self_):
                        raise ConnectionError("Supabase transient 503")
                return _FakeFlaky()
            return original_rpc(function_name, params)

        client.rpc = _flaky_rpc

        with patch("main.get_supabase_client", return_value=client), patch(
            "supabase_client.time.sleep"
        ):
            response = self.client.post(
                f"/admin/review/{VALID_POST_ID}/approve",
                headers=self._admin_headers(),
                json={"reason": "Looks acceptable."},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreaterEqual(call_count["n"], 2, "Expected at least one retry")
        self.assertEqual(
            client.rows["posts"][0]["moderation_status"], "ADMIN_APPROVED"
        )


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeSupabaseClient:
    def __init__(self, rows):
        self.rows = rows
        self.operations = []
        self.rpc_calls = []
        self.rpc_error = None

    def table(self, table_name):
        query = _FakeSupabaseQuery(self, table_name)
        self.operations.append(query)
        return query

    def rpc(self, function_name, params):
        query = _FakeRpcQuery(self, function_name, params)
        self.rpc_calls.append(query)
        return query


class _FakeSupabaseQuery:
    def __init__(self, client, table_name):
        self.client = client
        self.table_name = table_name
        self.select_columns = None
        self.insert_payload = None
        self.update_payload = None
        self.filters = []
        self.order_args = None
        self.limit_value = None
        self.range_args = None

    def select(self, columns):
        self.select_columns = columns
        return self

    def insert(self, payload):
        self.insert_payload = payload
        return self

    def update(self, payload):
        self.update_payload = payload
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column, values):
        self.filters.append(("in", column, values))
        return self

    def order(self, column, desc=False):
        self.order_args = (column, desc)
        return self

    def limit(self, count):
        self.limit_value = count
        return self

    def range(self, start, end):
        self.range_args = (start, end)
        return self

    def execute(self):
        if self.insert_payload is not None:
            row = dict(self.insert_payload)
            self.client.rows.setdefault(self.table_name, []).append(row)
            return _FakeResponse([row])

        rows = [
            row
            for row in self.client.rows.get(self.table_name, [])
            if self._matches(row)
        ]

        if self.order_args is not None:
            column, desc = self.order_args
            rows = sorted(rows, key=lambda row: row.get(column) or "", reverse=desc)

        if self.range_args is not None:
            start, end = self.range_args
            rows = rows[start : end + 1]

        if self.limit_value is not None:
            rows = rows[: self.limit_value]

        if self.update_payload is not None:
            for row in rows:
                row.update(self.update_payload)
            return _FakeResponse([dict(row) for row in rows])

        return _FakeResponse([dict(row) for row in rows])

    def _matches(self, row):
        for operator, column, value in self.filters:
            if operator == "eq" and row.get(column) != value:
                return False
            if operator == "in" and row.get(column) not in value:
                return False
        return True


class _FakeRpcQuery:
    def __init__(self, client, function_name, params):
        self.client = client
        self.function_name = function_name
        self.params = params

    def execute(self):
        if self.client.rpc_error is not None:
            raise self.client.rpc_error

        if self.function_name not in {"approve_review", "reject_review"}:
            return _FakeResponse([])

        action = "APPROVE" if self.function_name == "approve_review" else "REJECT"
        status_value = (
            "ADMIN_APPROVED" if self.function_name == "approve_review" else "ADMIN_REJECTED"
        )
        post_id = self.params["p_post_id"]
        posts = [row for row in self.client.rows.get("posts", []) if row.get("id") == post_id]
        if not posts:
            raise RuntimeError("Post was not found")
        review_jobs = [
            row
            for row in self.client.rows.get("moderation_jobs", [])
            if row.get("post_id") == post_id and row.get("status") == "UNDER_REVIEW"
        ]
        if not review_jobs:
            raise RuntimeError("No UNDER_REVIEW moderation job was found")

        posts[0]["moderation_status"] = status_value
        self.client.rows.setdefault("admin_review_actions", []).append(
            {
                "post_id": post_id,
                "admin_action": action,
                "reason": self.params.get("p_reason"),
                "admin_id": self.params.get("p_admin_id"),
                "ip_address": self.params.get("p_ip_address"),
                "request_id": self.params.get("p_request_id"),
            }
        )
        for job in review_jobs:
            job["status"] = "COMPLETED"
        return _FakeResponse([])
