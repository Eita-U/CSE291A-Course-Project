#!/usr/bin/env python3
"""Copy high-level memory-analysis artifacts into one folder.

Does not copy traces, 1 Hz telemetry, or full results.json.

  python3 scripts/bundle_memory_analysis.py --run-id phase1-fifo-005
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Compact files an agent needs for memory analysis (no request traces).
BUNDLE_FILES = (
    "run_manifest.json",
    "memory_analysis.json",
    "per_suite_memory.json",
    "per_suite_memory.csv",
    "per_suite_memory_aggregate.json",
    "per_suite_memory_aggregate.csv",
    "analysis_brief.md",
    "analysis_brief.json",
)

BUNDLE_BRIEF_TABLES = (
    "suite_medians.csv",
    "trial_health.csv",
    "per_trial_suites.csv",
    "isolated_context.csv",
    "isolated_response.csv",
    "isolated_steps.csv",
    "isolated_tool_delay.csv",
    "prefix_reuse.csv",
    "retained_context.csv",
    "oversize_context.csv",
)

README = """# Memory analysis bundle

High-level artifacts only. No request traces, no 1 Hz GPU CSV, no results.json.

## Setup
- `run_manifest.json` — model, concurrency, context/response budgets, timeouts

## Memory (from EC2 monitor, joined to suite windows)
- `memory_analysis.json` — overall GPU VRAM, server RSS, KV, prompt-cache peaks, coverage
- `per_suite_memory.csv` / `.json` — per trial × suite peaks
- `per_suite_memory_aggregate.csv` / `.json` — median peaks across trials

## Workload context (needed to interpret memory vs completed work)
- `analysis_brief.md` — bottleneck notes plus suite table with GPU/KV columns
- `analysis_brief.json` — same numbers as JSON
- `analysis_brief/suite_medians.csv` — throughput, latency, fail counts, GPU, KV
- `analysis_brief/trial_health.csv` — unexpected vs oversize failures
- other `analysis_brief/*.csv` — context, response, steps, tool delay, prefix, retained, oversize

## Not included (raw)
`gpu_system_metrics.csv`, `suite_boundary_events.jsonl`, `trial_*/*.trace.json`,
`results.json`, `summary.json`, `workload_manifest.json`, `graphs/`.
"""


def copy_if_exists(src: Path, dest: Path, copied: list, missing: list, out: Path):
    if not src.exists():
        missing.append(str(src))
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    copied.append(str(dest.relative_to(out)))


def main():
    parser = argparse.ArgumentParser(
        description="Bundle high-level memory analysis files for one microbenchmark run."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination folder (default: results/<run-id>/memory_analysis_bundle)",
    )
    args = parser.parse_args()
    root = args.results_root / args.run_id
    out = args.out or (root / "memory_analysis_bundle")
    out.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []

    for name in BUNDLE_FILES:
        copy_if_exists(root / name, out / name, copied, missing, out)

    brief_src = root / "analysis_brief"
    brief_dest = out / "analysis_brief"
    for name in BUNDLE_BRIEF_TABLES:
        copy_if_exists(brief_src / name, brief_dest / name, copied, missing, out)

    (out / "README.md").write_text(README, encoding="utf-8")
    copied.append("README.md")

    manifest = {
        "run_id": args.run_id,
        "source": str(root),
        "destination": str(out),
        "copied": copied,
        "missing": missing,
        "excluded": [
            "gpu_system_metrics.csv",
            "suite_boundary_events.jsonl",
            "trial_*/*.trace.json",
            "results.json",
            "summary.json",
            "graphs/",
        ],
    }
    (out / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"Bundled {len(copied)} files into {out}")
    if missing:
        print("Missing (generate with analyze_memory.py / summarize_microbenchmark.py):")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
