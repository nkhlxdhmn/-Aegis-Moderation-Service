"""Unit tests for the moderation queue service."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import queue_service


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.queries = []
        self.rpc_calls = []

    def table(self, table_name):
        query = FakeQuery(self, table_name)
        self.queries.append(query)
        return query

    def rpc(self, function_name, params):
        query = FakeRpcQuery(self, function_name, params)
        self.rpc_calls.append(query)
        return query


class FakeQuery:
    def __init__(self, client, table_name):
        self.client = client
        self.table_name = table_name
        self.insert_payload = None
        self.update_payload = None
        self.select_columns = None
        self.filters = []
        self.order_args = None
        self.limit_value = None

    def insert(self, payload):
        self.insert_payload = payload
        return self

    def update(self, payload):
        self.update_payload = payload
        return self

    def select(self, columns):
        self.select_columns = columns
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def order(self, column, desc=False):
        self.order_args = (column, desc)
        return self

    def limit(self, count):
        self.limit_value = count
        return self

    def execute(self):
        response = self.client.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)


class FakeRpcQuery:
    def __init__(self, client, function_name, params):
        self.client = client
        self.function_name = function_name
        self.params = params

    def execute(self):
        response = self.client.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(response)


class AtomicClaimClient:
    def __init__(self, pending_jobs):
        self.pending_jobs = list(pending_jobs)
        self.claimed_ids = set()
        self.rpc_calls = []

    def rpc(self, function_name, params):
        self.rpc_calls.append((function_name, params))
        return AtomicClaimQuery(self, params["p_worker_id"])


class AtomicClaimQuery:
    def __init__(self, client, worker_id):
        self.client = client
        self.worker_id = worker_id

    def execute(self):
        for job in self.client.pending_jobs:
            if job["id"] not in self.client.claimed_ids:
                self.client.claimed_ids.add(job["id"])
                claimed_job = {
                    **job,
                    "status": "PROCESSING",
                    "worker_id": self.worker_id,
                }
                return FakeResponse([claimed_job])
        return FakeResponse([])


class DueAwareClaimClient:
    def __init__(self, pending_jobs, now):
        self.pending_jobs = list(pending_jobs)
        self.now = now
        self.rpc_calls = []

    def rpc(self, function_name, params):
        self.rpc_calls.append((function_name, params))
        return DueAwareClaimQuery(self, params["p_worker_id"])


class DueAwareClaimQuery:
    def __init__(self, client, worker_id):
        self.client = client
        self.worker_id = worker_id

    def execute(self):
        for job in self.client.pending_jobs:
            next_attempt_at = job.get("next_attempt_at")
            if next_attempt_at:
                attempt_at = datetime.fromisoformat(next_attempt_at)
                if attempt_at > self.client.now:
                    continue
            return FakeResponse(
                [
                    {
                        **job,
                        "status": "PROCESSING",
                        "worker_id": self.worker_id,
                    }
                ]
            )
        return FakeResponse([])


class QueueServiceTests(TestCase):
    def test_create_job_inserts_pending_job(self):
        created_job = {
            "id": "job-1",
            "post_id": "post-1",
            "image_url": "https://example.com/image.jpg",
            "status": "PENDING",
            "retry_count": 0,
            "max_retries": 3,
        }
        client = FakeClient([[created_job]])

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.create_job(
                "post-1",
                "https://example.com/image.jpg",
            )

        self.assertEqual(result, created_job)
        self.assertEqual(client.queries[0].table_name, "moderation_jobs")
        payload = client.queries[0].insert_payload
        self.assertEqual(payload["post_id"], "post-1")
        self.assertEqual(payload["image_url"], "https://example.com/image.jpg")
        self.assertEqual(payload["status"], "PENDING")
        self.assertEqual(payload["retry_count"], 0)
        self.assertEqual(payload["max_retries"], 3)
        self.assertIn("next_attempt_at", payload)

    def test_get_next_pending_job_fetches_oldest_pending_job(self):
        pending_job = {"id": "job-1", "status": "PENDING"}
        client = FakeClient([[pending_job]])

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.get_next_pending_job()

        query = client.queries[0]
        self.assertEqual(result, pending_job)
        self.assertEqual(query.select_columns, "*")
        self.assertIn(("status", "PENDING"), query.filters)
        self.assertEqual(query.order_args, ("created_at", False))
        self.assertEqual(query.limit_value, 1)

    def test_get_next_pending_job_returns_none_when_empty(self):
        client = FakeClient([[]])

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.get_next_pending_job()

        self.assertIsNone(result)

    def test_claim_next_pending_job_uses_atomic_rpc(self):
        claimed_job = {
            "id": "job-1",
            "status": "PROCESSING",
            "worker_id": "worker-1",
        }
        client = FakeClient([[claimed_job]])

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.claim_next_pending_job("worker-1")

        self.assertEqual(result, claimed_job)
        self.assertEqual(client.rpc_calls[0].function_name, "claim_next_moderation_job")
        self.assertEqual(client.rpc_calls[0].params, {"p_worker_id": "worker-1"})

    def test_claim_next_pending_job_returns_none_when_empty(self):
        client = FakeClient([[]])

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.claim_next_pending_job("worker-1")

        self.assertIsNone(result)

    def test_two_workers_cannot_claim_same_job(self):
        client = AtomicClaimClient(
            [
                {
                    "id": "job-1",
                    "post_id": "post-1",
                    "image_url": "https://example.com/image.jpg",
                    "status": "PENDING",
                }
            ]
        )

        with patch("queue_service.get_supabase_client", return_value=client):
            first_claim = queue_service.claim_next_pending_job("worker-1")
            second_claim = queue_service.claim_next_pending_job("worker-2")

        self.assertIsNotNone(first_claim)
        self.assertEqual(first_claim["id"], "job-1")
        self.assertEqual(first_claim["worker_id"], "worker-1")
        self.assertIsNone(second_claim)

    def test_future_next_attempt_job_is_not_claimable(self):
        now = datetime.now(UTC)
        client = DueAwareClaimClient(
            [
                {
                    "id": "job-1",
                    "status": "PENDING",
                    "next_attempt_at": (now + timedelta(minutes=5)).isoformat(),
                }
            ],
            now,
        )

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.claim_next_pending_job("worker-1")

        self.assertIsNone(result)

    def test_due_next_attempt_job_is_claimable(self):
        now = datetime.now(UTC)
        client = DueAwareClaimClient(
            [
                {
                    "id": "job-1",
                    "status": "PENDING",
                    "next_attempt_at": (now - timedelta(seconds=1)).isoformat(),
                }
            ],
            now,
        )

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.claim_next_pending_job("worker-1")

        self.assertEqual(result["id"], "job-1")
        self.assertEqual(result["worker_id"], "worker-1")

    def test_retry_backoff_delays_are_one_five_and_fifteen_minutes(self):
        before = datetime.now(UTC)

        retry_1 = datetime.fromisoformat(queue_service.retry_backoff_until(1))
        retry_2 = datetime.fromisoformat(queue_service.retry_backoff_until(2))
        retry_3 = datetime.fromisoformat(queue_service.retry_backoff_until(3))

        self.assertGreaterEqual(retry_1, before + timedelta(seconds=55))
        self.assertLessEqual(retry_1, before + timedelta(seconds=65))
        self.assertGreaterEqual(retry_2, before + timedelta(minutes=4, seconds=55))
        self.assertLessEqual(retry_2, before + timedelta(minutes=5, seconds=5))
        self.assertGreaterEqual(retry_3, before + timedelta(minutes=14, seconds=55))
        self.assertLessEqual(retry_3, before + timedelta(minutes=15, seconds=5))

    def test_claim_rpc_migration_uses_skip_locked(self):
        migration_path = (
            Path(__file__).resolve().parents[2]
            / "supabase"
            / "migrations"
            / "202606130002_claim_next_moderation_job_rpc.sql"
        )

        sql = migration_path.read_text(encoding="utf-8").upper()

        self.assertIn("FOR UPDATE SKIP LOCKED", sql)
        self.assertIn("ORDER BY CREATED_AT ASC", sql)
        self.assertIn("STATUS = 'PROCESSING'", sql)

    def test_mark_processing_updates_status_worker_and_started_at(self):
        updated_job = {"id": "job-1", "status": "PROCESSING"}
        client = FakeClient([[updated_job]])

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.mark_processing("job-1", "worker-1")

        payload = client.queries[0].update_payload
        self.assertEqual(result, updated_job)
        self.assertEqual(payload["status"], "PROCESSING")
        self.assertEqual(payload["worker_id"], "worker-1")
        self.assertIn("started_at", payload)
        self.assertIn(("id", "job-1"), client.queries[0].filters)

    def test_mark_completed_updates_status_and_completed_at(self):
        updated_job = {"id": "job-1", "status": "COMPLETED"}
        client = FakeClient([[updated_job]])

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.mark_completed("job-1")

        payload = client.queries[0].update_payload
        self.assertEqual(result, updated_job)
        self.assertEqual(payload["status"], "COMPLETED")
        self.assertIn("completed_at", payload)

    def test_mark_under_review_updates_status_and_completed_at(self):
        updated_job = {"id": "job-1", "status": "UNDER_REVIEW"}
        client = FakeClient([[updated_job]])

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.mark_under_review("job-1")

        payload = client.queries[0].update_payload
        self.assertEqual(result, updated_job)
        self.assertEqual(payload["status"], "UNDER_REVIEW")
        self.assertIn("completed_at", payload)

    def test_mark_failed_updates_status_error_and_completed_at(self):
        updated_job = {"id": "job-1", "status": "FAILED"}
        client = FakeClient([[updated_job]])

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.mark_failed("job-1", "download failed")

        payload = client.queries[0].update_payload
        self.assertEqual(result, updated_job)
        self.assertEqual(payload["status"], "FAILED")
        self.assertEqual(payload["error_message"], "download failed")
        self.assertIn("completed_at", payload)

    def test_increment_retry_returns_updated_retry_count(self):
        client = FakeClient([[{"retry_count": 2}], [{"retry_count": 3}]])

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.increment_retry("job-1")

        self.assertEqual(result, 3)
        self.assertEqual(client.queries[0].select_columns, "retry_count")
        self.assertIn(("id", "job-1"), client.queries[0].filters)
        self.assertEqual(client.queries[1].update_payload, {"retry_count": 3})
        self.assertIn(("id", "job-1"), client.queries[1].filters)

    def test_supabase_exception_raises_queue_service_error(self):
        client = FakeClient([RuntimeError("db unavailable")])

        with patch("queue_service.get_supabase_client", return_value=client):
            with self.assertRaises(queue_service.QueueServiceError):
                queue_service.get_next_pending_job()

    def test_get_queue_depth_by_status_uses_metrics_rpc(self):
        client = FakeClient(
            [
                [
                    {"status": "PENDING", "count": 15},
                    {"status": "PROCESSING", "count": 2},
                    {"status": "UNDER_REVIEW", "count": 3},
                    {"status": "FAILED", "count": 1},
                ]
            ]
        )

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.get_queue_depth_by_status()

        self.assertEqual(result["PENDING"], 15)
        self.assertEqual(result["PROCESSING"], 2)
        self.assertEqual(result["UNDER_REVIEW"], 3)
        self.assertEqual(result["FAILED"], 1)
        self.assertEqual(client.rpc_calls[0].function_name, "get_queue_metrics")
        self.assertEqual(client.queries, [])

    def test_get_decision_counts_today_uses_aggregation_rpc(self):
        client = FakeClient([[{"approved_today": 125, "rejected_today": 17}]])

        with patch("queue_service.get_supabase_client", return_value=client):
            result = queue_service.get_decision_counts_today()

        self.assertEqual(result, {"approved_today": 125, "rejected_today": 17})
        self.assertEqual(
            client.rpc_calls[0].function_name,
            "get_moderation_decision_counts_today",
        )
        self.assertEqual(client.queries, [])
