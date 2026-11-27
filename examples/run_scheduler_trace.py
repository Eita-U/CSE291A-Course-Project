"""Replay agent traces through FIFO or slot-affinity scheduling.

Start llama-server first (inference_setup/start_server.sh) with PARALLEL equal
to --slots. By default this replays the same 20 sessions as the Phase 1
concurrency experiment. Results go to
measurement/results/phase2/<run_id>/results.json in the Phase 1 format, plus
the scheduler config and /metrics counters from before and after the run.

    python -m examples.run_scheduler_trace --scheduler affinity --slots 2
    python -m examples.run_scheduler_trace --scheduler fifo --slots 2

--synthetic replays small generated sessions instead (no trace file needed),
for a quick check in minutes rather than an hour.
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

import pandas as pd

from measurement.replay_trace import replay_trace
from measurement.synthetic_sessions import synthetic_sessions
from measurement.trace_loader import group_tasks, load_trace
from scheduler import FIFOScheduler, SlotAffinityScheduler
from scheduler.qwen_backend import QwenHTTPBackend

PHASE1_TASKS = "measurement/results/concurrency/concurrency-1-gap-1.0-cache-True-concurrency/results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scheduler", choices=("fifo", "affinity"), required=True)
    parser.add_argument("--slots", type=int, required=True,
                        help="Server --parallel; FIFO concurrency or affinity slot count")
    parser.add_argument("--hold-s", type=float, default=2.0,
                        help="Affinity: seconds to hold a slot for its agent's next step")
    parser.add_argument("--max-skips", type=int, default=10,
                        help="Affinity: skips before a request takes the next free slot")
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--trace", default="data/processed/trace_32k.parquet")
    parser.add_argument("--tasks-from", default=PHASE1_TASKS,
                        help="Results file whose task IDs select the sessions to replay")
    parser.add_argument("--synthetic", action="store_true",
                        help="Replay generated sessions instead of the trace")
    parser.add_argument("--agents", type=int, default=8, help="Synthetic: number of agents")
    parser.add_argument("--steps", type=int, default=8, help="Synthetic: steps per agent")
    parser.add_argument("--words-per-step", type=int, default=1000,
                        help="Synthetic: words of new context per step")
    parser.add_argument("--pre-gap-scale", type=float, default=1.0)
    parser.add_argument("--no-cache-prompt", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout (s)")
    parser.add_argument("--run-id", help="Output folder name (default from the settings)")
    return parser.parse_args()


def read_server_metrics(server: str) -> dict[str, float]:
    text = httpx.get(f"{server}/metrics", timeout=10).text
    values = {}
    for line in text.splitlines():
        if line.startswith("llamacpp:"):
            name, _, value = line[len("llamacpp:"):].partition(" ")
            values[name] = float(value)
    return values


def check_server_slots(server: str, slots: int) -> None:
    response = httpx.get(f"{server}/slots", timeout=10)
    response.raise_for_status()
    n_server = len(response.json())
    if n_server != slots:
        raise SystemExit(f"--slots {slots} does not match the server's {n_server} slots (PARALLEL)")


def summarize(results: dict, metrics_before: dict, metrics_after: dict) -> dict:
    requests = [r for task in results["tasks"].values() for r in task["requests"]]
    tasks_ok = sum(task["status"] == "success" for task in results["tasks"].values())
    queue_ms = sorted(r["timing_metadata"]["queue_time_ms"] for r in requests)

    def delta(name):
        return metrics_after.get(name, 0.0) - metrics_before.get(name, 0.0)

    cached, computed = delta("prompt_tokens_cached_total"), delta("prompt_tokens_total")
    summary = {
        "tasks_succeeded": f"{tasks_ok}/{len(results['tasks'])}",
        "requests": len(requests),
        "throughput_tasks_per_min": tasks_ok / (results["timing_metadata"]["run_duration_ms"] / 60000),
        "mean_queue_time_s": sum(queue_ms) / len(queue_ms) / 1000 if queue_ms else 0.0,
        "p95_queue_time_s": queue_ms[int(0.95 * (len(queue_ms) - 1))] / 1000 if queue_ms else 0.0,
        "prompt_tokens_cached": cached,
        "prompt_tokens_computed": computed,
        "cache_reuse_rate": cached / (cached + computed) if cached + computed else 0.0,
        "prompt_cache_ram_restores": delta("prompt_cache_loads_total"),
    }
    if requests and "slot_hit" in requests[0]:
        summary["slot_hit_rate"] = sum(r["slot_hit"] for r in requests) / len(requests)
        summary["max_skips"] = max(r["skips"] for r in requests)
    return summary


async def main() -> None:
    args = parse_args()
    cache_prompt = not args.no_cache_prompt
    server = args.server.rstrip("/")

    if args.synthetic:
        sessions = synthetic_sessions(args.agents, args.steps, args.words_per_step)
        tasks = {sid: pd.DataFrame(rows) for sid, rows in sessions.items()}
    else:
        task_ids = list(json.loads(Path(args.tasks_from).read_text(encoding="utf-8"))["tasks"])
        tasks = group_tasks(load_trace(args.trace))
        missing = [t for t in task_ids if t not in tasks]
        if missing:
            raise SystemExit(f"{len(missing)} sessions from {args.tasks_from} are not in {args.trace}")
        tasks = {t: tasks[t] for t in task_ids}

    if args.scheduler == "affinity":
        check_server_slots(server, args.slots)
    metrics_before = read_server_metrics(server)

    async with QwenHTTPBackend(f"{server}/v1/chat/completions", api_style="openai", timeout_s=args.timeout) as backend:
        if args.scheduler == "fifo":
            scheduler = FIFOScheduler(backend, concurrency=args.slots, request_timeout_s=args.timeout)
        else:
            scheduler = SlotAffinityScheduler(
                backend, slots=args.slots, request_timeout_s=args.timeout,
                hold_s=args.hold_s, max_skips=args.max_skips,
            )
        async with scheduler:
            start_utc = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            task_results = await replay_trace(scheduler, tasks, cache_prompt, args.pre_gap_scale)
            duration_ms = (time.perf_counter() - started) * 1000
            finish_utc = datetime.now(timezone.utc).isoformat()

    metrics_after = read_server_metrics(server)
    config = {
        "scheduler": args.scheduler, "concurrency": args.slots, "slots": args.slots,
        "cache_prompt": cache_prompt, "pre_gap_scale": args.pre_gap_scale,
        "request_timeout_s": args.timeout, "n_tasks": len(tasks),
    }
    if args.synthetic:
        config.update(workload="synthetic", agents=args.agents, steps=args.steps,
                      words_per_step=args.words_per_step)
    else:
        config.update(workload="trace", trace=args.trace, tasks_from=args.tasks_from)
    if args.scheduler == "affinity":
        config.update(hold_s=args.hold_s, max_skips=args.max_skips)
    results = {
        "config": config,
        "timing_metadata": {
            "run_start_time": start_utc,
            "run_finish_time": finish_utc,
            "run_duration_ms": duration_ms,
        },
        "server_metrics": {"before": metrics_before, "after": metrics_after},
        "tasks": task_results,
    }
    results["summary"] = summarize(results, metrics_before, metrics_after)

    run_id = args.run_id or (
        ("synthetic-" if args.synthetic else "")
        + f"{args.scheduler}-slots-{args.slots}"
        + (f"-hold-{args.hold_s}-skips-{args.max_skips}" if args.scheduler == "affinity" else "")
    )
    output = Path(f"measurement/results/phase2/{run_id}/results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results["summary"], indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    asyncio.run(main())
