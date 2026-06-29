"""Unit tests for Supabase moderation persistence."""

from copy import deepcopy
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import supabase_client


POST_ID = "11111111-1111-4111-8111-111111111111"


class SupabaseClientTests(TestCase):
    def test_write_results_success_writes_result_and_post_atomically(self):
        client = _AtomicRpcClient(
            {
                "posts": [{"id": POST_ID, "moderation_status": "PENDING"}],
                "moderation_results": [],
            }
        )

        with patch("supabase_client.get_supabase_client", return_value=client):
            supabase_client.write_results(
                post_id=POST_ID,
                scores={
                    "adult_score": 0.1,
                    "heritage_score": 0.8,
                    "child_safety_score": 0.0,
                    "violence_self_harm_score": 0.0,
                    "content_quality_score": 0.2,
                },
                category="Religious & Spiritual Heritage",
                confidence=0.92,
                decision="APPROVED",
                reason="APPROVED: Looks safe.",
                ocr_text="temple text",
            )

        self.assertEqual(len(client.rows["moderation_results"]), 1)
        self.assertEqual(client.rows["moderation_results"][0]["post_id"], POST_ID)
        self.assertEqual(client.rows["moderation_results"][0]["decision"], "APPROVED")
        self.assertEqual(client.rows["posts"][0]["moderation_status"], "APPROVED")
        self.assertEqual(
            client.rows["posts"][0]["category_name"],
            "Religious & Spiritual Heritage",
        )
        self.assertEqual(client.rpc_calls[0].function_name, "persist_moderation_result")

    def test_write_results_failure_writes_neither_table(self):
        client = _AtomicRpcClient(
            {
                "posts": [],
                "moderation_results": [],
            }
        )

        with patch("supabase_client.get_supabase_client", return_value=client):
            with self.assertRaises(RuntimeError):
                supabase_client.write_results(
                    post_id=POST_ID,
                    scores={"adult_score": 0.1},
                    category="Education & Documentation",
                    confidence=0.8,
                    decision="APPROVED",
                    reason="APPROVED: Looks safe.",
                    ocr_text="",
                )

        self.assertEqual(client.rows["posts"], [])
        self.assertEqual(client.rows["moderation_results"], [])

    def test_write_results_propagates_rpc_errors(self):
        client = _AtomicRpcClient(
            {
                "posts": [{"id": POST_ID, "moderation_status": "PENDING"}],
                "moderation_results": [],
            },
            rpc_error=RuntimeError("rpc unavailable"),
        )

        with patch("supabase_client.get_supabase_client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "rpc unavailable"):
                supabase_client.write_results(
                    post_id=POST_ID,
                    scores={},
                    category="Education & Documentation",
                    confidence=0.8,
                    decision="APPROVED",
                    reason="APPROVED: Looks safe.",
                    ocr_text="",
                )

        self.assertEqual(client.rows["posts"][0]["moderation_status"], "PENDING")
        self.assertEqual(client.rows["moderation_results"], [])

    def test_persist_moderation_result_migration_is_atomic_rpc(self):
        migration_path = (
            Path(__file__).resolve().parents[2]
            / "supabase"
            / "migrations"
            / "202606150001_persist_moderation_result_rpc.sql"
        )

        sql = migration_path.read_text(encoding="utf-8").upper()

        self.assertIn("CREATE OR REPLACE FUNCTION PUBLIC.PERSIST_MODERATION_RESULT", sql)
        self.assertIn("INSERT INTO PUBLIC.MODERATION_RESULTS", sql)
        self.assertIn("UPDATE PUBLIC.POSTS", sql)
        self.assertIn("GET DIAGNOSTICS UPDATED_POST_COUNT = ROW_COUNT", sql)
        self.assertIn("RAISE EXCEPTION", sql)
        self.assertIn("GRANT EXECUTE", sql)


class _AtomicRpcClient:
    def __init__(self, rows, rpc_error=None):
        self.rows = deepcopy(rows)
        self.rpc_error = rpc_error
        self.rpc_calls = []

    def rpc(self, function_name, params):
        call = _AtomicRpcQuery(self, function_name, params)
        self.rpc_calls.append(call)
        return call

    def table(self, table_name):
        raise AssertionError(f"Unexpected direct table write to {table_name}")


class _AtomicRpcQuery:
    def __init__(self, client, function_name, params):
        self.client = client
        self.function_name = function_name
        self.params = params

    def execute(self):
        if self.client.rpc_error is not None:
            raise self.client.rpc_error
        if self.function_name != "persist_moderation_result":
            raise RuntimeError(f"Unexpected RPC {self.function_name}")

        snapshot = deepcopy(self.client.rows)
        try:
            self.client.rows.setdefault("moderation_results", []).append(
                {
                    "post_id": self.params["p_post_id"],
                    "adult_score": self.params["p_adult_score"],
                    "heritage_score": self.params["p_heritage_score"],
                    "child_safety_score": self.params["p_child_safety_score"],
                    "violence_self_harm_score": self.params[
                        "p_violence_self_harm_score"
                    ],
                    "content_quality_score": self.params[
                        "p_content_quality_score"
                    ],
                    "decision": self.params["p_decision"],
                    "reason": self.params["p_reason"],
                }
            )

            updated = False
            for post in self.client.rows.setdefault("posts", []):
                if post.get("id") == self.params["p_post_id"]:
                    post.update(
                        {
                            "moderation_status": self.params["p_decision"],
                            "category_name": self.params["p_category_name"],
                            "category_confidence": self.params[
                                "p_category_confidence"
                            ],
                            "ocr_text": self.params["p_ocr_text"],
                        }
                    )
                    updated = True

            if not updated:
                raise RuntimeError("Post was not found")
        except Exception:
            self.client.rows = snapshot
            raise

        return _RpcResponse()


class _RpcResponse:
    data = None
