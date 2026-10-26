import json
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
