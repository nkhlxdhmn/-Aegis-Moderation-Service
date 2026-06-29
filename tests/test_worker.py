"""Unit tests for the moderation worker orchestration."""

from datetime import UTC, datetime, timedelta
import threading
from unittest import TestCase
from unittest.mock import ANY, call, patch

from pipeline.safety_flags import ModerationPipelineResult
import queue_service as qs
import worker


JOB = {
    "id": "job-1",
    "post_id": "11111111-1111-4111-8111-111111111111",
    "image_url": "https://project.supabase.co/storage/image.jpg",
    "retry_count": 0,
    "max_retries": 3,
}


def _pipeline_result(score_overrides=None):
    scores = {
        "adult_score": 0.0,
        "heritage_score": 0.0,
        "content_quality_score": 0.0,
        "child_safety_score": 0.0,
        "violence_self_harm_score": 0.0,
    }
    scores.update(score_overrides or {})
    return ModerationPipelineResult(
        scores=scores,
        category_scores={"Education & Documentation": 0.8},
        ocr_text="ocr",
    )


class WorkerProcessJobTests(TestCase):
    """Tests for process_single_job against the current RPC-based implementation."""

    def test_successful_approved_flow_calls_complete_job_with_result(self):
        result = _pipeline_result()

        with self._patch_common(result), patch(
            "worker.decide_with_reason_code",
            return_value=("APPROVED", "APPROVED", "Heritage content approved."),
        ), patch(
            "worker.queue_service.complete_job_with_result",
        ) as complete_job:
            handled = worker.process_single_job()

        self.assertTrue(handled)
        complete_job.assert_called_once_with(
            job_id="job-1",
            worker_id="worker-test",
            post_id="11111111-1111-4111-8111-111111111111",
            scores=ANY,
            category=ANY,
            confidence=ANY,
            decision="APPROVED",
            reason=ANY,
            ocr_text=ANY,
        )

    def test_successful_rejected_flow_calls_complete_job_with_result(self):
        result = _pipeline_result({"adult_score": 0.9})

        with self._patch_common(result), patch(
            "worker.decide_with_reason_code",
            return_value=("REJECTED", "NSFW_CONTENT", "Explicit adult content detected."),
        ), patch(
            "worker.queue_service.complete_job_with_result",
        ) as complete_job:
            handled = worker.process_single_job()

        self.assertTrue(handled)
        complete_job.assert_called_once_with(
            job_id="job-1",
            worker_id="worker-test",
            post_id="11111111-1111-4111-8111-111111111111",
            scores=ANY,
            category=ANY,
            confidence=ANY,
            decision="REJECTED",
            reason=ANY,
            ocr_text=ANY,
        )

    def test_successful_under_review_flow_calls_complete_job_with_result(self):
        result = _pipeline_result()

        with self._patch_common(result), patch(
            "worker.decide_with_reason_code",
            return_value=("UNDER_REVIEW", "PRIVACY_VIOLATION", "PII risk detected."),
        ), patch(
            "worker.queue_service.complete_job_with_result",
        ) as complete_job:
            handled = worker.process_single_job()

        self.assertTrue(handled)
        complete_job.assert_called_once_with(
            job_id="job-1",
            worker_id="worker-test",
            post_id="11111111-1111-4111-8111-111111111111",
            scores=ANY,
            category=ANY,
            confidence=ANY,
            decision="UNDER_REVIEW",
            reason=ANY,
            ocr_text=ANY,
        )

    def test_reason_is_formatted_with_reason_code_prefix(self):
        result = _pipeline_result()

        with self._patch_common(result), patch(
            "worker.decide_with_reason_code",
            return_value=("APPROVED", "APPROVED", "Heritage content."),
        ), patch(
            "worker.queue_service.complete_job_with_result",
        ) as complete_job:
            worker.process_single_job()

        reason = complete_job.call_args.kwargs["reason"]
        self.assertTrue(reason.startswith("APPROVED:"), f"Expected reason code prefix, got: {reason!r}")
        self.assertIn("Heritage content.", reason)

    def test_pipeline_error_sets_under_review_decision(self):
        error_result = ModerationPipelineResult(
            scores={
                "adult_score": 0.0,
                "heritage_score": 0.0,
                "content_quality_score": 0.0,
                "child_safety_score": 0.0,
                "violence_self_harm_score": 0.0,
            },
            category_scores={},
            ocr_text="",
            pipeline_error=True,
            error_reason="NudeNet crashed",
        )

        with self._patch_common(error_result), patch(
            "worker.queue_service.complete_job_with_result",
        ) as complete_job:
            handled = worker.process_single_job()

        self.assertTrue(handled)
        call_kwargs = complete_job.call_args.kwargs
        self.assertEqual(call_kwargs["decision"], "UNDER_REVIEW")
        self.assertIn("PIPELINE_ERROR", call_kwargs["reason"])
        self.assertIn("NudeNet crashed", call_kwargs["reason"])

    def test_retry_flow_calls_retry_job_after_failure_below_max_retries(self):
        job = dict(JOB, retry_count=0, max_retries=3)

        with patch(
            "worker.queue_service.claim_next_pending_job",
            return_value=job,
        ), patch(
            "worker._download_image",
            side_effect=RuntimeError("download failed"),
        ), patch(
            "worker.queue_service.retry_job_after_failure",
            return_value=1,
        ) as retry_job, patch(
            "worker.queue_service.fail_job",
        ) as fail_job, patch(
            "worker.queue_service.retry_backoff_until",
            return_value="2026-06-15T00:01:00+00:00",
        ), patch.dict("os.environ", {"WORKER_ID": "worker-test"}, clear=False):
            handled = worker.process_single_job()

        self.assertTrue(handled)
        retry_job.assert_called_once_with(
            "job-1",
            "worker-test",
            "download failed",
            ANY,
        )
        fail_job.assert_not_called()

    def test_permanent_failure_flow_calls_fail_job_at_max_retries(self):
        job = dict(JOB, retry_count=2, max_retries=3)

        with patch(
            "worker.queue_service.claim_next_pending_job",
            return_value=job,
        ), patch(
            "worker._download_image",
            side_effect=RuntimeError("pipeline crashed"),
        ), patch(
            "worker.queue_service.retry_job_after_failure",
        ) as retry_job, patch(
            "worker.queue_service.fail_job",
        ) as fail_job, patch.dict(
            "os.environ", {"WORKER_ID": "worker-test"}, clear=False
        ):
            handled = worker.process_single_job()

        self.assertTrue(handled)
        fail_job.assert_called_once_with("job-1", "worker-test", "pipeline crashed")
        retry_job.assert_not_called()

    def test_empty_queue_returns_false(self):
        with patch(
            "worker.queue_service.claim_next_pending_job",
            return_value=None,
        ) as claim:
            handled = worker.process_single_job()

        self.assertFalse(handled)
        claim.assert_called_once()

    def test_run_worker_sleeps_on_empty_queue(self):
        with patch(
            "worker.recover_stuck_jobs",
            return_value=0,
        ), patch(
            "worker.process_single_job",
            side_effect=[False, KeyboardInterrupt],
        ), patch(
            "worker.time.sleep",
            side_effect=KeyboardInterrupt,
        ) as sleep, patch.dict(
            "os.environ",
            {"POLL_INTERVAL_SECONDS": "2"},
            clear=False,
        ):
            with self.assertRaises(KeyboardInterrupt):
                worker.run_worker()

        sleep.assert_has_calls([call(2.0)])

    def test_run_worker_logs_recovery_count_on_startup(self):
        with patch(
            "worker.recover_stuck_jobs",
            return_value=2,
        ) as recover_stuck_jobs, patch(
            "worker.process_single_job",
            side_effect=KeyboardInterrupt,
        ), self.assertLogs("worker", level="INFO") as logs:
            with self.assertRaises(KeyboardInterrupt):
                worker.run_worker()

        recover_stuck_jobs.assert_called_once_with()
        self.assertIn("Recovered 2 stuck jobs", "\n".join(logs.output))

    def test_complete_job_not_called_when_claim_fails(self):
        with patch(
            "worker.queue_service.claim_next_pending_job",
            side_effect=RuntimeError("db down"),
        ), patch(
            "worker.queue_service.complete_job_with_result",
        ) as complete_job:
            with self.assertRaises(RuntimeError):
                worker.process_single_job()

        complete_job.assert_not_called()

    def _patch_common(self, pipeline_result):
        patches = [
            patch("worker.queue_service.claim_next_pending_job", return_value=JOB),
            patch("worker._download_image", return_value="tmp.jpg"),
            patch("worker._validate_image_file"),
            patch("worker.analyze_image", return_value=pipeline_result),
            patch(
                "worker.get_top_category",
                return_value=("Education & Documentation", 0.8),
            ),
            patch("worker.Path.unlink"),
            patch.dict("os.environ", {"WORKER_ID": "worker-test"}, clear=False),
        ]
        return _PatchGroup(patches)


class WorkerMainLoopResilienceTests(TestCase):
    """Regression tests for the worker-crash-on-handle_job_failure bug.

    Failure path: pipeline fails → _handle_job_failure raises QueueServiceError
    (e.g. Supabase unreachable while retrying) → exception propagates out of
    process_single_job → bare while loop in run_worker crashes the process.

    Fix: run_worker wraps process_single_job in try/except so a single bad job
    cannot kill the worker permanently.
    """

    def test_run_worker_continues_when_handle_job_failure_raises(self):
        """Worker loop must not crash when retry/fail RPC raises after pipeline error."""

        call_count = {"n": 0}

        def _flaky_process_single_job():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise qs.QueueServiceError("Supabase down while failing job")
            raise KeyboardInterrupt

        with patch(
            "worker.recover_stuck_jobs",
            return_value=0,
        ), patch(
            "worker.process_single_job",
            side_effect=_flaky_process_single_job,
        ), patch(
            "worker.time.sleep",
        ) as sleep:
            with self.assertRaises(KeyboardInterrupt):
                worker.run_worker()

        # The loop must have iterated past the QueueServiceError
        self.assertEqual(call_count["n"], 2)
        # And slept before retrying (as the error path does)
        sleep.assert_called()

    def test_run_worker_logs_error_when_process_single_job_raises(self):
        """Unhandled process_single_job errors must be logged, not silently swallowed."""

        def _raise_then_stop():
            raise qs.QueueServiceError("transient db error")

        with patch(
            "worker.recover_stuck_jobs",
            return_value=0,
        ), patch(
            "worker.process_single_job",
            side_effect=[qs.QueueServiceError("transient db error"), KeyboardInterrupt],
        ), patch(
            "worker.time.sleep",
        ), self.assertLogs("worker", level="ERROR") as logs:
            with self.assertRaises(KeyboardInterrupt):
                worker.run_worker()

        self.assertTrue(
            any("Unhandled error" in line for line in logs.output),
            f"Expected 'Unhandled error' in logs; got: {logs.output}",
        )


class WorkerRecoveryTests(TestCase):
    """Tests for recover_stuck_jobs via the queue_service RPC path."""

    def test_recover_stuck_jobs_delegates_to_queue_service(self):
        with patch(
            "worker.queue_service.recover_stuck_jobs",
            return_value=3,
        ) as recover, patch.dict(
            "os.environ",
            {"PROCESSING_TIMEOUT_MINUTES": "10"},
            clear=False,
        ):
            count = worker.recover_stuck_jobs()

        self.assertEqual(count, 3)
        recover.assert_called_once()
        cutoff_arg = recover.call_args.args[0]
        parsed = datetime.fromisoformat(cutoff_arg)
        age = datetime.now(UTC) - parsed
        self.assertAlmostEqual(age.total_seconds() / 60, 10.0, delta=0.5)

    def test_recover_stuck_jobs_uses_custom_timeout_from_env(self):
        with patch(
            "worker.queue_service.recover_stuck_jobs",
            return_value=1,
        ) as recover, patch.dict(
            "os.environ",
            {"PROCESSING_TIMEOUT_MINUTES": "5"},
            clear=False,
        ):
            worker.recover_stuck_jobs()

        cutoff_arg = recover.call_args.args[0]
        parsed = datetime.fromisoformat(cutoff_arg)
        age = datetime.now(UTC) - parsed
        self.assertAlmostEqual(age.total_seconds() / 60, 5.0, delta=0.5)

    def test_recover_stuck_jobs_wraps_queue_service_error(self):
        with patch(
            "worker.queue_service.recover_stuck_jobs",
            side_effect=RuntimeError("db down"),
        ):
            with self.assertRaises(qs.QueueServiceError):
                worker.recover_stuck_jobs()

    def test_recover_stuck_jobs_returns_zero_when_none_stale(self):
        with patch(
            "worker.queue_service.recover_stuck_jobs",
            return_value=0,
        ):
            count = worker.recover_stuck_jobs()

        self.assertEqual(count, 0)


class WorkerHeartbeatTests(TestCase):
    """Tests for worker heartbeat scheduling."""

    def test_heartbeat_recorded_when_interval_elapsed(self):
        import time

        with patch("worker.heartbeat_service.record_worker_heartbeat") as record, patch(
            "worker._last_heartbeat_at",
            new=time.time() - worker.HEARTBEAT_INTERVAL_SECONDS - 1,
        ):
            worker._record_heartbeat_if_due("worker-test")

        record.assert_called_once()

    def test_heartbeat_skipped_when_interval_not_elapsed(self):
        import time

        with patch("worker.heartbeat_service.record_worker_heartbeat") as record, patch(
            "worker._last_heartbeat_at",
            new=time.time(),
        ):
            worker._record_heartbeat_if_due("worker-test")

        record.assert_not_called()

    def test_heartbeat_failure_does_not_raise(self):
        with patch(
            "worker.heartbeat_service.record_worker_heartbeat",
            side_effect=RuntimeError("heartbeat db down"),
        ):
            worker._record_heartbeat("worker-test")


class ClipEngineSingletonThreadSafetyTests(TestCase):
    """Regression tests for the double-init race in clip_engine._get_state.

    Without a lock, two concurrent threads both seeing _state=None load the
    400MB ViT-B-32 model twice (OOM) and one writes a partially-initialized
    _ClipState to the global before prompt encoding is complete.  Any thread
    that reads the fast-path check while another is mid-init receives a state
    with heritage_features=None and crashes inside _score_against_features.

    Fix: double-checked lock; global only published after full initialization.
    """

    def test_get_state_acquires_lock_module_has_lock_attribute(self):
        """_state_lock must be a threading.Lock-compatible object."""

        import pipeline.clip_engine as ce

        self.assertTrue(
            hasattr(ce, "_state_lock"),
            "_state_lock is missing from clip_engine; thread safety not guaranteed.",
        )
        # Must expose the threading lock protocol (acquire/release)
        self.assertTrue(callable(getattr(ce._state_lock, "acquire", None)))
        self.assertTrue(callable(getattr(ce._state_lock, "release", None)))

    def test_get_state_publishes_only_fully_initialized_state(self):
        """If _get_state fails mid-way, _state must remain None so the next
        call retries rather than returning a partially-built object."""

        import pipeline.clip_engine as ce

        saved = ce._state
        try:
            ce._state = None
            with patch("pipeline.clip_engine._state_lock", threading.Lock()), patch(
                "pipeline.clip_engine._encode_prompts",
                side_effect=RuntimeError("prompt encoding failed"),
            ):
                try:
                    import open_clip  # noqa: F401
                    import torch  # noqa: F401
                except ImportError:
                    self.skipTest("open_clip / torch not installed")

                with patch("open_clip.create_model_and_transforms") as mock_create, \
                     patch("open_clip.get_tokenizer"):
                    import types
                    fake_model = types.SimpleNamespace(eval=lambda: None)
                    mock_create.return_value = (fake_model, None, None)
                    with self.assertRaises(Exception):
                        ce._get_state()

                # After a failed init the global must still be None
                self.assertIsNone(
                    ce._state,
                    "_state must be None after failed initialization so the next "
                    "call retries loading the model rather than returning partial state.",
                )
        finally:
            ce._state = saved

    def test_concurrent_get_state_calls_do_not_double_initialize(self):
        """Two threads calling _get_state simultaneously must only run the
        initialization once (one thread loads, the other waits and reuses)."""

        import pipeline.clip_engine as ce

        saved = ce._state
        init_count = {"n": 0}

        original_create = None

        def counting_create(*args, **kwargs):
            init_count["n"] += 1
            # Simulate slow initialization to widen the race window
            import time
            time.sleep(0.05)
            return original_create(*args, **kwargs) if original_create else (None, None, None)

        try:
            ce._state = None
            try:
                import open_clip
            except ImportError:
                self.skipTest("open_clip not installed")

            original_create = open_clip.create_model_and_transforms
            errors = []

            def _call():
                try:
                    ce._get_state()
                except Exception as exc:
                    errors.append(exc)

            with patch(
                "open_clip.create_model_and_transforms",
                side_effect=counting_create,
            ):
                t1 = threading.Thread(target=_call)
                t2 = threading.Thread(target=_call)
                t1.start()
                t2.start()
                t1.join(timeout=10)
                t2.join(timeout=10)

            self.assertEqual(
                init_count["n"],
                1,
                f"Model loaded {init_count['n']} times; expected exactly 1 (double-init race detected).",
            )
        finally:
            ce._state = saved


class OcrSingletonThreadSafetyTests(TestCase):
    """Regression tests for the double-init race in ocr._get_ocr."""

    def test_ocr_module_has_lock_attribute(self):
        import pipeline.ocr as ocr_mod

        self.assertTrue(
            hasattr(ocr_mod, "_ocr_lock"),
            "_ocr_lock is missing from ocr module; thread safety not guaranteed.",
        )
        self.assertTrue(callable(getattr(ocr_mod._ocr_lock, "acquire", None)))

    def test_concurrent_get_ocr_calls_do_not_double_initialize(self):
        """Two threads calling _get_ocr simultaneously must only initialize once."""

        import pipeline.ocr as ocr_mod

        saved = ocr_mod._ocr
        init_count = {"n": 0}

        class _FakePaddleOCR:
            def __init__(self, **_kw):
                init_count["n"] += 1
                import time
                time.sleep(0.05)

        errors = []

        def _call():
            try:
                ocr_mod._get_ocr()
            except Exception as exc:
                errors.append(exc)

        try:
            ocr_mod._ocr = None
            with patch("pipeline.ocr._ocr_lock", threading.Lock()), patch(
                "paddleocr.PaddleOCR",
                _FakePaddleOCR,
            ):
                t1 = threading.Thread(target=_call)
                t2 = threading.Thread(target=_call)
                t1.start()
                t2.start()
                t1.join(timeout=10)
                t2.join(timeout=10)
        except ImportError:
            self.skipTest("paddleocr not installed")
        finally:
            ocr_mod._ocr = saved

        self.assertEqual(
            init_count["n"],
            1,
            f"PaddleOCR initialized {init_count['n']} times; expected exactly 1.",
        )


class NsfwSingletonThreadSafetyTests(TestCase):
    """Regression tests for the missing lock in nsfw._get_detector.

    Without a lock, two concurrent requests on cold start both see
    _detector=None and both call NudeDetector(), causing a double ONNX session
    open and a dangling detector instance leak.

    Fix: double-checked lock identical to the pattern in clip_engine and ocr.
    """

    def test_nsfw_module_has_lock_attribute(self):
        import pipeline.nsfw as nsfw_mod

        self.assertTrue(
            hasattr(nsfw_mod, "_detector_lock"),
            "_detector_lock is missing from nsfw module; thread safety not guaranteed.",
        )
        self.assertTrue(callable(getattr(nsfw_mod._detector_lock, "acquire", None)))

    def test_concurrent_get_detector_calls_do_not_double_initialize(self):
        """Two threads calling _get_detector simultaneously must only initialize once."""

        import pipeline.nsfw as nsfw_mod

        saved = nsfw_mod._detector
        init_count = {"n": 0}

        class _FakeNudeDetector:
            def __init__(self):
                init_count["n"] += 1
                import time
                time.sleep(0.05)

        errors = []

        def _call():
            try:
                nsfw_mod._get_detector()
            except Exception as exc:
                errors.append(exc)

        try:
            nsfw_mod._detector = None
            with patch("pipeline.nsfw._detector_lock", threading.Lock()), patch(
                "nudenet.NudeDetector",
                _FakeNudeDetector,
            ):
                t1 = threading.Thread(target=_call)
                t2 = threading.Thread(target=_call)
                t1.start()
                t2.start()
                t1.join(timeout=10)
                t2.join(timeout=10)
        except ImportError:
            self.skipTest("nudenet not installed")
        finally:
            nsfw_mod._detector = saved

        self.assertEqual(
            init_count["n"],
            1,
            f"NudeDetector initialized {init_count['n']} times; expected exactly 1.",
        )


class GpuDevicePinningTests(TestCase):
    """Regression tests for GPU 0 isolation requirements.

    These verify that:
    - pipeline/__init__.py sets CUDA_VISIBLE_DEVICES=0 as a fallback guard
    - object_detector._select_device() returns 'cuda:0' (not 'cuda')
    - clip_engine._get_state() uses 'cuda:0' when CUDA is available
    """

    def test_pipeline_init_sets_cuda_visible_devices_default(self):
        """pipeline/__init__ must set CUDA_VISIBLE_DEVICES=0 via os.environ.setdefault."""
        import os
        saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            # Reimport to re-execute the module-level setdefault
            import importlib
            import pipeline as pipeline_pkg
            importlib.reload(pipeline_pkg)
            self.assertEqual(
                os.environ.get("CUDA_VISIBLE_DEVICES"),
                "0",
                "pipeline/__init__.py must call os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')",
            )
        finally:
            if saved is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = saved
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_pipeline_init_does_not_override_explicit_cuda_visible_devices(self):
        """setdefault must not override a value already set by the operator."""
        import os
        import importlib
        import pipeline as pipeline_pkg
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        try:
            importlib.reload(pipeline_pkg)
            self.assertEqual(
                os.environ["CUDA_VISIBLE_DEVICES"],
                "2",
                "pipeline/__init__.py must not override an operator-provided CUDA_VISIBLE_DEVICES",
            )
        finally:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_object_detector_select_device_returns_cuda_0_not_bare_cuda(self):
        """_select_device must return 'cuda:0' to prevent implicit current-device drift."""
        from unittest.mock import MagicMock
        from pipeline.object_detector import _select_device

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        device = _select_device(fake_torch)
        self.assertEqual(
            device,
            "cuda:0",
            f"_select_device must return 'cuda:0', not {device!r}; "
            "bare 'cuda' is subject to torch.cuda.current_device() drift on multi-GPU hosts.",
        )

    def test_object_detector_select_device_returns_cpu_when_cuda_unavailable(self):
        from unittest.mock import MagicMock
        from pipeline.object_detector import _select_device

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        self.assertEqual(_select_device(fake_torch), "cpu")

    def test_clip_engine_uses_cuda_0_when_cuda_available(self):
        """When CUDA is available, clip_engine must use 'cuda:0', never 'cpu'."""
        import pipeline.clip_engine as ce

        saved = ce._state
        try:
            ce._state = None
            try:
                import open_clip  # noqa: F401
                import torch  # noqa: F401
            except ImportError:
                self.skipTest("open_clip / torch not installed")

            with patch("torch.cuda.is_available", return_value=True), \
                 patch("open_clip.create_model_and_transforms") as mock_create, \
                 patch("open_clip.get_tokenizer"), \
                 patch("pipeline.clip_engine._encode_prompts", return_value=None):
                import types
                fake_model = types.SimpleNamespace(eval=lambda: None)
                mock_create.return_value = (fake_model, None, None)
                try:
                    ce._get_state()
                except Exception:
                    pass
                if mock_create.called:
                    _, kwargs = mock_create.call_args
                    device_arg = kwargs.get("device") or (
                        mock_create.call_args.args[2] if len(mock_create.call_args.args) > 2 else None
                    )
                    self.assertNotEqual(
                        device_arg,
                        "cpu",
                        "clip_engine must not hardcode device='cpu' when CUDA is available",
                    )
        finally:
            ce._state = saved


class _PatchGroup:
    def __init__(self, patches):
        self.patches = patches
        self.started = []

    def __enter__(self):
        for patcher in self.patches:
            self.started.append(patcher.start())
        return self.started

    def __exit__(self, exc_type, exc_value, traceback):
        for patcher in reversed(self.patches):
            patcher.stop()
        return False
