import json
import tempfile
import unittest
from pathlib import Path

from examples.run_microbenchmarks import build_config, parse_args, qwen_backend_factory
from workload.benchmarks import percentile, quick_config, run_microbenchmarks


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


if __name__ == "__main__":
    unittest.main()
