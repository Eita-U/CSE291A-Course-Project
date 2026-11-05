#!/usr/bin/env python3
"""Collapse a Phase 1 run into assignment-sized tables.

Reads trial metrics/traces plus optional memory analysis outputs, then writes
a compact brief an agent can use instead of raw JSONL/CSV.

  python3 scripts/summarize_microbenchmark.py --run-id phase1-fifo-004
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

MEASURED_SUITES = (
    "isolated_context",
    "isolated_response",
    "isolated_steps",
    "isolated_tool_delay",
    "oversize_context",
    "prefix_reuse",
    "retained_context",
    "mixed_matrix",
    "concurrency_1",
    "concurrency_2",
    "concurrency_4",
)


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def median(xs):
    present = [x for x in xs if x is not None]
    return statistics.median(present) if present else None


def round_or_none(value, ndigits=3):
    if value is None:
        return None
    return round(float(value), ndigits)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def get_suite(trial, name):
    for suite in trial["suites"]:
        if suite["name"] == name:
            return suite
    return None


def suite_row(trial_id, suite):
    metrics = suite["metrics"]
    statuses = metrics.get("request_status_counts") or {}
    n_failed = metrics.get("n_failed_requests", 0) or 0
    expected_fail = suite["name"] == "oversize_context"
    healthy = expected_fail or n_failed == 0
    return {
        "trial_id": trial_id,
        "suite": suite["name"],
        "concurrency": suite.get("concurrency"),
        "healthy": healthy,
        "n_requests": metrics["n_requests"],
        "n_tasks": metrics["n_tasks"],
        "n_success_requests": statuses.get("success", 0),
        "n_failed_requests": n_failed,
        "n_timeout_requests": metrics.get("n_timeout_requests", 0),
        "n_success_tasks": metrics.get("n_success_tasks", 0),
        "success_task_fraction": metrics.get("success_task_fraction"),
        "req_per_s": metrics["throughput_requests_per_s"],
        "task_per_s": metrics["throughput_tasks_per_s"],
        "request_p50_ms": metrics["request_total_ms"]["p50"],
        "request_p95_ms": metrics["request_total_ms"]["p95"],
        "queue_p50_ms": metrics["request_queue_ms"]["p50"],
        "queue_p95_ms": metrics["request_queue_ms"]["p95"],
        "api_p50_ms": metrics["request_api_ms"]["p50"],
        "api_p95_ms": metrics["request_api_ms"]["p95"],
        "task_p50_ms": metrics["task_completion_ms"]["p50"],
        "task_p95_ms": metrics["task_completion_ms"]["p95"],
        "client_rss_mib": suite.get("peak_benchmark_process_rss_mib"),
    }


NUMERIC_KEYS = [
    "n_requests", "n_tasks", "n_success_requests", "n_success_tasks",
    "success_task_fraction", "req_per_s", "task_per_s", "request_p50_ms",
    "request_p95_ms", "queue_p50_ms", "queue_p95_ms", "api_p50_ms",
    "api_p95_ms", "task_p50_ms", "task_p95_ms", "client_rss_mib",
]


def digits_for(key):
    if key in {"n_requests", "n_tasks", "n_success_requests",
               "n_failed_requests", "n_timeout_requests", "n_success_tasks"}:
        return 0
    if "per_s" in key or key == "success_task_fraction":
        return 4
    return 1


def median_suite_table(rows, names):
    by_name = defaultdict(list)
    for row in rows:
        by_name[row["suite"]].append(row)
    table = []
    for name in names:
        group = by_name.get(name, [])
        if not group:
            continue
        healthy = [r for r in group if r["healthy"]]
        source = healthy or group
        out = {
            "suite": name,
            "concurrency": group[0]["concurrency"],
            "n_trials": len(group),
            "n_healthy_trials": len(healthy),
            "healthy_trial_ids": [r["trial_id"] for r in healthy],
            "n_failed_requests_all_trials": sum(r["n_failed_requests"] for r in group),
            "n_timeout_requests_all_trials": sum(r["n_timeout_requests"] for r in group),
            "latency_from": "healthy_trials" if healthy else "all_trials_unhealthy",
        }
        for key in NUMERIC_KEYS:
            out[key] = round_or_none(median(r[key] for r in source), digits_for(key))
        table.append(out)
    return table


def class_key(request, suite):
    if suite == "prefix_reuse":
        return request.get("prefix_mode") or request["profile_name"]
    if suite == "retained_context":
        return request.get("context_mode") or request["profile_name"]
    if suite == "isolated_context":
        return "long" if request.get("prompt_token_estimate", 0) >= 400 else "short"
    if suite == "isolated_response":
        return f"max_tokens_{request.get('max_tokens')}"
    if suite == "isolated_steps":
        planned = 3 if "3step" in request.get("profile_name", "") else 1
        return f"{planned}_step"
    if suite == "isolated_tool_delay":
        delay = request.get("tool_delay_s") or 0.0
        return "tool_50ms" if delay > 0 else "tool_0ms"
    if suite == "oversize_context":
        return request.get("status", "unknown")
    return request.get("profile_name")


def profile_table(root: Path, suite: str, healthy_trial_ids):
    per_class = defaultdict(lambda: defaultdict(list))
    task_class = defaultdict(lambda: defaultdict(list))
    trial_ids = healthy_trial_ids or []
    for trial_id in trial_ids:
        path = root / f"trial_{trial_id}" / f"{suite}.trace.json"
        if not path.exists():
            continue
        trace = load_json(path)
        for request in trace.get("requests", []):
            key = class_key(request, suite)
            timing = request.get("timing_metadata") or {}
            usage = request.get("token_usage") or {}
            per_class[key]["api_ms"].append(timing.get("api_time_ms"))
            per_class[key]["total_ms"].append(timing.get("total_time_ms"))
            per_class[key]["queue_ms"].append(timing.get("queue_time_ms"))
            per_class[key]["prompt_tokens"].append(usage.get("prompt_tokens"))
            per_class[key]["completion_tokens"].append(usage.get("completion_tokens"))
            per_class[key]["status"].append(request.get("status"))
        for task in trace.get("tasks", []):
            key = task.get("prefix_mode") or task.get("context_mode")
            if suite not in {"prefix_reuse", "retained_context"}:
                name = task.get("profile_name", "")
                if suite == "isolated_steps":
                    key = "3_step" if "3step" in name else "1_step"
                elif suite == "isolated_tool_delay":
                    key = "tool_50ms" if "tool" in name and "notool" not in name else "tool_0ms"
                elif suite == "isolated_context":
                    key = "long" if "long_ctx" in name else "short"
                elif suite == "isolated_response":
                    key = "max_tokens_256" if "long_resp" in name else "max_tokens_32"
                else:
                    key = name
            task_class[key]["task_ms"].append(task.get("task_duration_ms"))
            task_class[key]["status"].append(task.get("status"))

    rows = []
    for key in sorted(per_class):
        block = per_class[key]
        apis = [x for x in block["api_ms"] if x is not None]
        prompts = [x for x in block["prompt_tokens"] if x is not None]
        completions = [x for x in block["completion_tokens"] if x is not None]
        statuses = block["status"]
        task_block = task_class.get(key, {})
        task_ms = [x for x in task_block.get("task_ms", []) if x is not None]
        rows.append({
            "suite": suite,
            "class": key,
            "n_requests": len(statuses),
            "n_success_requests": sum(1 for s in statuses if s == "success"),
            "n_failed_requests": sum(1 for s in statuses if s != "success"),
            "api_p50_ms": round_or_none(percentile(apis, 50), 1),
            "api_p95_ms": round_or_none(percentile(apis, 95), 1),
            "queue_p50_ms": round_or_none(
                percentile([x for x in block["queue_ms"] if x is not None], 50), 1
            ),
            "request_p50_ms": round_or_none(
                percentile([x for x in block["total_ms"] if x is not None], 50), 1
            ),
            "task_p50_ms": round_or_none(percentile(task_ms, 50), 1),
            "prompt_tokens_min": min(prompts) if prompts else None,
            "prompt_tokens_max": max(prompts) if prompts else None,
            "completion_tokens_min": min(completions) if completions else None,
            "completion_tokens_max": max(completions) if completions else None,
        })
    return rows


def join_memory(suite_table, memory_by_suite):
    for row in suite_table:
        mem = memory_by_suite.get(row["suite"], {})
        row.update({
            "peak_gpu_vram_mib": mem.get("median_peak_gpu_vram_mib"),
            "peak_server_gpu_mib": mem.get("median_peak_server_gpu_memory_mib"),
            "peak_server_rss_mib": mem.get("median_peak_server_rss_mib"),
            "peak_kv_cache_tokens": mem.get("median_peak_kv_cache_tokens"),
            "peak_kv_usage_ratio": mem.get("median_peak_kv_cache_usage_ratio"),
            "peak_prompt_cache_mib": mem.get("median_peak_prompt_cache_mib"),
            "prompt_cache_hit_fraction": mem.get("median_prompt_cache_hit_fraction"),
            "peak_requests_processing": mem.get("median_peak_requests_processing"),
            "peak_requests_deferred": mem.get("median_peak_requests_deferred"),
            "memory_trials_covered": mem.get("trials_with_telemetry"),
        })
    return suite_table


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def findings(brief):
    conc = {row["suite"]: row for row in brief["tables"]["suite_medians"]
            if row["suite"].startswith("concurrency_")}
    prefix = {row["class"]: row for row in brief["tables"]["prefix_reuse"]}
    retained = {row["class"]: row for row in brief["tables"]["retained_context"]}
    context = {row["class"]: row for row in brief["tables"]["isolated_context"]}
    response = {row["class"]: row for row in brief["tables"]["isolated_response"]}
    tool = {row["class"]: row for row in brief["tables"]["isolated_tool_delay"]}
    oversize = brief["tables"]["oversize_context"]
    mem = brief.get("memory_overall") or {}

    c1, c2, c4 = conc.get("concurrency_1"), conc.get("concurrency_2"), conc.get("concurrency_4")
    notes = []
    health = brief.get("trial_health") or []
    unhealthy = [h for h in health if h["unexpected_failed_requests"] > 0]
    if unhealthy:
        notes.append(
            "Unexpected failures (not oversize admission): "
            + "; ".join(
                f"trial {h['trial_id']} {h['unexpected_failed_requests']} failed requests "
                f"in {h['unhealthy_suites']}"
                for h in unhealthy
            )
            + ". Latency/throughput tables use only healthy trial/suite pairs."
        )
    if c1 and c2 and c4 and c1["n_healthy_trials"] and c2["n_healthy_trials"] and c4["n_healthy_trials"]:
        gain_12 = c2["req_per_s"] / c1["req_per_s"] if c1["req_per_s"] else None
        gain_24 = c4["req_per_s"] / c2["req_per_s"] if c2["req_per_s"] else None
        notes.append(
            "Concurrency 1→2 raises request throughput "
            f"{round_or_none(gain_12, 2)}× while 2→4 is "
            f"{round_or_none(gain_24, 2)}× "
            f"(healthy trials {c1['healthy_trial_ids']}/{c2['healthy_trial_ids']}/"
            f"{c4['healthy_trial_ids']}). API p50 goes "
            f"{c2['api_p50_ms']}→{c4['api_p50_ms']} ms at conc 4, "
            "so extra workers wait on llama.cpp slots rather than the FIFO queue."
        )
    timeouts = sum(row.get("n_timeout_requests_all_trials") or 0
                   for row in brief["tables"]["suite_medians"])
    if oversize and oversize[0]["n_success_requests"] == 0:
        notes.append(
            "oversize_context is 0% success on every trial: HTTP admission reject "
            "when the prompt exceeds the 2048-token slot. "
            f"Timeouts across suites: {int(timeouts)}."
        )
    if context.get("short") and context.get("long"):
        notes.append(
            "Long vs short context API p50 is "
            f"{context['long']['api_p50_ms']} vs {context['short']['api_p50_ms']} ms "
            "at max_tokens=32; decode length dominates once prompts fit."
        )
    if response.get("max_tokens_32") and response.get("max_tokens_256"):
        notes.append(
            "Long vs short response API p50 is "
            f"{response['max_tokens_256']['api_p50_ms']} vs "
            f"{response['max_tokens_32']['api_p50_ms']} ms."
        )
    if tool.get("tool_0ms") and tool.get("tool_50ms"):
        notes.append(
            "50 ms tool idle is much smaller than GPU/queue time "
            f"(task p50 {tool['tool_50ms']['task_p50_ms']} vs "
            f"{tool['tool_0ms']['task_p50_ms']} ms in the same suite)."
        )
    if prefix.get("shared") and prefix.get("unique"):
        notes.append(
            "Shared-prefix API p50 "
            f"{prefix['shared']['api_p50_ms']} ms vs unique "
            f"{prefix['unique']['api_p50_ms']} ms. Timing favors reuse; "
            "this is not a llama.cpp cache-hit counter."
        )
    if retained.get("retained") and retained.get("fresh"):
        notes.append(
            "Retained vs length-matched fresh API p50 "
            f"{retained['retained']['api_p50_ms']} vs "
            f"{retained['fresh']['api_p50_ms']} ms (essentially matched)."
        )
    gpu = mem.get("peak_gpu_vram_mib")
    kv = mem.get("peak_kv_cache_tokens")
    if gpu is not None:
        notes.append(
            f"Peak device VRAM {gpu} MiB with peak KV {kv} / "
            f"{mem.get('kv_cache_tokens_max')} tokens. VRAM is nearly flat; "
            "the binding memory limit is per-slot context (2048 tokens), not L4 used-memory growth."
        )
    bottleneck = (
        "FIFO queueing of GPU decode behind llama.cpp --parallel 2; "
        "oversize prompts fail admission rather than OOM."
    )
    return {
        "dominant_bottleneck": bottleneck,
        "observations": notes,
        "assignment_coverage": {
            "task_completion_time": "suite and class task_p50_ms / task_p95_ms",
            "queueing_time": "queue_p50_ms / queue_p95_ms",
            "tail_latency": "request_p95_ms and api_p95_ms",
            "throughput": "req_per_s and task_per_s vs concurrency",
            "peak_memory": "GPU VRAM, server RSS, KV tokens, prompt cache, client RSS",
            "failures_timeouts": "n_failed_requests / n_timeout_requests plus oversize class",
            "concurrency": "concurrency_1/2/4 table",
            "context_growth": "isolated_context + oversize_context",
            "idle_periods": "isolated_tool_delay",
            "retained_vs_prefix": "retained_context and prefix_reuse class tables",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()
    root = args.results_root / args.run_id

    manifest = load_json(root / "run_manifest.json")
    n_trials = int(manifest.get("benchmark", {}).get("n_trials") or manifest.get("n_trials") or 3)

    trials = []
    for trial_id in range(1, n_trials + 1):
        path = root / f"trial_{trial_id}" / "results.json"
        if path.exists():
            trials.append(load_json(path))

    suite_rows = []
    for trial in trials:
        trial_id = trial.get("trial_id")
        for suite in trial.get("suites", []):
            if suite["name"] in MEASURED_SUITES:
                suite_rows.append(suite_row(trial_id, suite))

    memory_by_suite = {}
    mem_agg_path = root / "per_suite_memory_aggregate.json"
    mem_overall = None
    if mem_agg_path.exists():
        for row in load_json(mem_agg_path):
            memory_by_suite[row["suite"]] = row
    if (root / "memory_analysis.json").exists():
        analysis = load_json(root / "memory_analysis.json")
        mem_overall = {
            **analysis.get("overall_peaks", {}),
            "baseline": analysis.get("initial_post_load_baseline"),
            "monitor": analysis.get("monitor"),
            "coverage": analysis.get("benchmark"),
        }

    suite_medians = join_memory(
        median_suite_table(suite_rows, MEASURED_SUITES),
        memory_by_suite,
    )
    healthy_ids = {
        row["suite"]: row["healthy_trial_ids"] for row in suite_medians
    }
    trial_health = []
    by_trial = defaultdict(list)
    for row in suite_rows:
        by_trial[row["trial_id"]].append(row)
    for trial_id, group in sorted(by_trial.items()):
        unexpected = [r for r in group if not r["healthy"]]
        trial_health.append({
            "trial_id": trial_id,
            "suites": len(group),
            "healthy_suites": len(group) - len(unexpected),
            "unhealthy_suites": [r["suite"] for r in unexpected],
            "unexpected_failed_requests": sum(r["n_failed_requests"] for r in unexpected),
        })

    brief = {
        "run_id": args.run_id,
        "note": (
            "Assignment-sized summary. Prefer this file over raw traces. "
            "Suite latency/throughput are medians of healthy trials only "
            "(oversize_context is expected-fail). Class tables use those "
            "same healthy trials. Failures are summed across all trials."
        ),
        "trial_health": trial_health,
        "setup": {
            "trials": n_trials,
            "seed": manifest.get("seed"),
            "backend": manifest.get("backend"),
            "concurrency_levels": manifest.get("concurrency_levels"),
            "context_tokens": manifest.get("benchmark", {}).get("context_tokens"),
            "response_tokens": manifest.get("benchmark", {}).get("response_tokens"),
            "oversize_context_tokens": manifest.get("benchmark", {}).get(
                "oversize_context_tokens"
            ),
            "model": manifest.get("benchmark", {}).get("model"),
            "git_revision": manifest.get("git_revision"),
        },
        "memory_overall": mem_overall,
        "tables": {
            "suite_medians": suite_medians,
            "isolated_context": profile_table(
                root, "isolated_context", healthy_ids["isolated_context"]
            ),
            "isolated_response": profile_table(
                root, "isolated_response", healthy_ids["isolated_response"]
            ),
            "isolated_steps": profile_table(
                root, "isolated_steps", healthy_ids["isolated_steps"]
            ),
            "isolated_tool_delay": profile_table(
                root, "isolated_tool_delay", healthy_ids["isolated_tool_delay"]
            ),
            "prefix_reuse": profile_table(
                root, "prefix_reuse", healthy_ids["prefix_reuse"]
            ),
            "retained_context": profile_table(
                root, "retained_context", healthy_ids["retained_context"]
            ),
            "oversize_context": profile_table(
                root, "oversize_context", healthy_ids["oversize_context"]
            ),
        },
    }
    brief["findings"] = findings(brief)

    out_dir = root / "analysis_brief"
    out_dir.mkdir(exist_ok=True)
    (root / "analysis_brief.json").write_text(
        json.dumps(brief, indent=2), encoding="utf-8"
    )
    write_csv(out_dir / "suite_medians.csv", suite_medians)
    write_csv(out_dir / "trial_health.csv", trial_health)
    write_csv(out_dir / "per_trial_suites.csv", suite_rows)
    for name in (
        "isolated_context", "isolated_response", "isolated_steps",
        "isolated_tool_delay", "prefix_reuse", "retained_context",
        "oversize_context",
    ):
        write_csv(out_dir / f"{name}.csv", brief["tables"][name])

    lines = [
        f"# Analysis brief: {args.run_id}",
        "",
        brief["note"],
        "",
        "## Dominant bottleneck",
        brief["findings"]["dominant_bottleneck"],
        "",
        "## Observations",
    ]
    for item in brief["findings"]["observations"]:
        lines.append(f"- {item}")
    lines += [
        "",
        "## Suite medians (see analysis_brief/suite_medians.csv)",
        "",
        "| Suite | Healthy trials | Conc | Req/s | Task p50 ms | Queue p50 ms | API p50 ms | Request p95 ms | Fail (all trials) | GPU MiB | KV tok |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in suite_medians:
        lines.append(
            f"| {row['suite']} | {row['n_healthy_trials']}/{row['n_trials']} | "
            f"{row['concurrency']} | {row['req_per_s']} | "
            f"{row['task_p50_ms']} | {row['queue_p50_ms']} | {row['api_p50_ms']} | "
            f"{row['request_p95_ms']} | {row['n_failed_requests_all_trials']} | "
            f"{row.get('peak_gpu_vram_mib')} | {row.get('peak_kv_cache_tokens')} |"
        )
    (root / "analysis_brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {root / 'analysis_brief.json'}")
    print(f"Wrote {root / 'analysis_brief.md'}")
    print(f"Wrote tables under {out_dir}")


if __name__ == "__main__":
    main()
