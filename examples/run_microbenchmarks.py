"""Run every Task 3 microbenchmark and write traces plus graphs.

From the repository root:

    python3 -m examples.run_microbenchmarks
    python3 -m examples.run_microbenchmarks --quick
    python3 -m examples.run_microbenchmarks --out results/microbenchmarks

Uses the mock backend by default so this can run without a GPU. Swap in a
real LLMBackend factory when the inference server is available. The FIFO
scheduler is not modified.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from workload.benchmarks import BenchConfig, quick_config, run_microbenchmarks
from workload.plots import generate_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FIFO microbenchmarks and plot results")
    parser.add_argument("--out", type=Path, default=Path("results/microbenchmarks"))
    parser.add_argument("--quick", action="store_true", help="Smaller token budgets and fewer tasks")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    config = (
        quick_config(args.out) if args.quick
        else BenchConfig(out_dir=args.out, seed=args.seed)
    )
    report = await run_microbenchmarks(config)
    paths = generate_plots(report, args.out / "graphs")
    summary = {
        "out_dir": str(args.out.resolve()),
        "suites": [
            {
                "name": suite["name"],
                "concurrency": suite["concurrency"],
                "n_requests": suite["metrics"]["n_requests"],
                "n_tasks": suite["metrics"]["n_tasks"],
                "throughput_requests_per_s": suite["metrics"]["throughput_requests_per_s"],
                "request_total_ms_p95": suite["metrics"]["request_total_ms"]["p95"],
                "task_completion_ms_p95": suite["metrics"]["task_completion_ms"]["p95"],
                "peak_rss_mib": suite["peak_rss_bytes"] / (1024 * 1024),
            }
            for suite in report["suites"]
        ],
        "graphs": [str(path) for path in paths],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
