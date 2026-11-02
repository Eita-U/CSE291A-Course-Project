import json
import tempfile
import unittest
from pathlib import Path

from examples.run_microbenchmarks import build_config, parse_args, qwen_backend_factory
from workload.benchmarks import (
    BenchConfig,
    oversize_context_profiles,
    percentile,
    plan_suites,
    quick_config,
    run_microbenchmarks,
    summarize,
    token_maps,
)
from workload.definitions import LONG_CONTEXT_TOKENS, OVERSIZE_CONTEXT_TOKENS
from workload.instrumentation import InstrumentationRecorder


class PercentileTests(unittest.TestCase):
    def test_endpoints_and_interpolation(self):
        self.assertIsNone(percentile([], 50))
        self.assertEqual(percentile([3], 95), 3)
        self.assertEqual(percentile([0, 10], 0), 0)
        self.assertEqual(percentile([0, 10], 100), 10)
        self.assertEqual(percentile([0, 10], 50), 5)


class MicrobenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_quick_suites_write_traces(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            matplotlib = None
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report = await run_microbenchmarks(quick_config(out))
            names = [suite["name"] for suite in report["suites"]]
            self.assertIn("isolated_context", names)
            self.assertIn("isolated_response", names)
            self.assertIn("isolated_steps", names)
            self.assertIn("isolated_tool_delay", names)
            self.assertIn("oversize_context", names)
            self.assertIn("mixed_matrix", names)
            self.assertIn("concurrency_1", names)
            self.assertTrue((out / "results.json").exists())
            mixed = next(s for s in report["suites"] if s["name"] == "mixed_matrix")
            self.assertGreater(mixed["metrics"]["n_requests"], 0)
            self.assertGreater(mixed["metrics"]["throughput_requests_per_s"], 0)
            json.loads((out / "results.json").read_text())
            if matplotlib is not None:
                from workload.plots import generate_plots
                paths = generate_plots(report, out / "graphs")
                stems = {path.stem for path in paths}
                self.assertIn("mixed_request_latency", stems)
                self.assertIn("concurrency_throughput", stems)
                self.assertTrue((out / "graphs" / "index.html").exists())

    async def test_backend_aclose_after_each_suite(self):
        instances = []

        class TrackingBackend:
            def __init__(self):
                self.closed = False
                self.calls = 0

            async def generate(self, request):
                self.calls += 1
                return {
                    "response": "ok",
                    "token_usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }

            async def aclose(self):
                self.closed = True

        def factory():
            backend = TrackingBackend()
            instances.append(backend)
            return backend

        with tempfile.TemporaryDirectory() as tmp:
            await run_microbenchmarks(quick_config(Path(tmp)), backend_factory=factory)
        self.assertGreaterEqual(len(instances), 7)
        self.assertTrue(all(backend.closed for backend in instances))
        self.assertTrue(all(backend.calls > 0 for backend in instances))

    async def test_qwen_factory_through_mocked_http(self):
        import httpx

        seen_prompts = []

        def responder(request):
            body = json.loads(request.content)
            seen_prompts.append(body["messages"][0]["content"])
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "Answer"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
            factory = qwen_backend_factory(
                endpoint="http://example.test/v1/chat/completions",
                api_style="openai",
                model=None,
                timeout_s=5.0,
                disable_think=True,
                client=client,
            )
            with tempfile.TemporaryDirectory() as tmp:
                report = await run_microbenchmarks(
                    quick_config(Path(tmp), backend_label="qwen"),
                    backend_factory=factory,
                )
        mixed = next(s for s in report["suites"] if s["name"] == "mixed_matrix")
        self.assertEqual(
            mixed["metrics"]["request_status_counts"].get("success"),
            mixed["metrics"]["n_requests"],
        )
        self.assertTrue(seen_prompts)
        self.assertTrue(all("/no_think" in prompt for prompt in seen_prompts))


class ContextBudgetTests(unittest.TestCase):
    def test_default_long_context_fits_slot_not_oversize(self):
        ctx, _ = token_maps(BenchConfig(out_dir=Path(".")))
        self.assertEqual(ctx["long"], LONG_CONTEXT_TOKENS)
        self.assertEqual(LONG_CONTEXT_TOKENS, 512)
        mixed = next(plan for plan in plan_suites(BenchConfig(out_dir=Path("."))) if plan.name == "mixed_matrix")
        longs = [profile for profile in mixed.profiles if profile.context_class == "long"]
        self.assertTrue(longs)
        self.assertTrue(all(profile.context_tokens == 512 for profile in longs))

    def test_oversize_suite_not_folded_into_mixed(self):
        plans = plan_suites(BenchConfig(out_dir=Path(".")))
        self.assertIn("oversize_context", [plan.name for plan in plans])
        oversize = next(plan for plan in plans if plan.name == "oversize_context")
        mixed = next(plan for plan in plans if plan.name == "mixed_matrix")
        self.assertEqual(oversize.profiles[0].context_tokens, OVERSIZE_CONTEXT_TOKENS)
        self.assertTrue(all(profile.context_tokens != OVERSIZE_CONTEXT_TOKENS for profile in mixed.profiles))
        self.assertLessEqual(oversize.tasks_per_profile, 4)
        self.assertEqual(oversize_context_profiles(BenchConfig(out_dir=Path(".")))[0].name, "oversize_context_admission")

    def test_successful_only_excludes_failures(self):
        recorder = InstrumentationRecorder()
        timing = {
            "queue_time_ms": 10.0, "api_time_ms": 20.0, "total_time_ms": 30.0,
            "queue_entry_time": "t0", "execution_start_time": "t1", "execution_finish_time": "t2",
        }
        recorder.record_request(
            profile_name="ok",
            scheduler_response={
                "request_id": "ok-0", "task_id": "ok", "step_id": 0, "status": "success",
                "worker_id": 1, "error": None, "timing_metadata": timing,
                "token_usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
            prompt_token_estimate=10, max_tokens=8, tool_delay_s=0.0, tool_wait_ms=0.0,
        )
        recorder.record_request(
            profile_name="bad",
            scheduler_response={
                "request_id": "bad-0", "task_id": "bad", "step_id": 0, "status": "failed",
                "worker_id": 1, "error": "HTTP 400", "timing_metadata": {
                    **timing, "total_time_ms": 5.0, "api_time_ms": 5.0, "queue_time_ms": 0.0,
                },
                "token_usage": {},
            },
            prompt_token_estimate=2048, max_tokens=8, tool_delay_s=0.0, tool_wait_ms=0.0,
        )
        recorder.record_task(
            spec_task_id="ok", profile_name="ok", n_steps_planned=1,
            request_records=[recorder.requests[0]],
            task_start_time="t0", task_finish_time="t2", task_duration_ms=30.0,
        )
        recorder.record_task(
            spec_task_id="bad", profile_name="bad", n_steps_planned=1,
            request_records=[recorder.requests[1]],
            task_start_time="t0", task_finish_time="t2", task_duration_ms=5.0,
        )
        metrics = summarize(recorder, wall_time_s=1.0)
        self.assertEqual(metrics["n_requests"], 2)
        self.assertEqual(metrics["n_failed_requests"], 1)
        self.assertEqual(metrics["successful_only"]["n_requests"], 1)
        self.assertEqual(metrics["successful_only"]["n_tasks"], 1)
        self.assertEqual(metrics["successful_only"]["request_total_ms"]["mean"], 30.0)
        self.assertEqual(metrics["request_total_ms"]["n"], 2)


class MicrobenchmarkCliTests(unittest.TestCase):
    def test_qwen_flags_set_backend_and_timeout(self):
        args = parse_args(["--backend", "qwen", "--quick", "--out", "results/qwen"])
        config = build_config(args)
        self.assertEqual(config.backend_label, "qwen")
        self.assertEqual(config.request_timeout_s, 180.0)
        self.assertEqual(args.endpoint, "http://127.0.0.1:8080/v1/chat/completions")

    def test_mock_quick_keeps_short_timeout(self):
        args = parse_args(["--quick"])
        config = build_config(args)
        self.assertEqual(config.backend_label, "mock")
        self.assertEqual(config.request_timeout_s, 5.0)
        self.assertIsNone(config.api_style)

    def test_long_and_oversize_cli_overrides(self):
        args = parse_args([
            "--long-context-tokens", "640",
            "--oversize-context-tokens", "1800",
        ])
        config = build_config(args)
        self.assertEqual(dict(config.context_tokens)["long"], 640)
        self.assertEqual(config.oversize_context_tokens, 1800)

    def test_qwen_config_accepts_api_style(self):
        args = parse_args(["--backend", "qwen", "--quick", "--api-style", "openai"])
        config = build_config(args)
        self.assertEqual(config.api_style, "openai")


if __name__ == "__main__":
    unittest.main()
