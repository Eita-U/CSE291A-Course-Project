"""Run the Task 3 microbenchmark suites against the existing FIFO scheduler."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scheduler import FIFOScheduler

from .definitions import (
    LONG_CONTEXT_TOKENS,
    OVERSIZE_CONTEXT_TOKENS,
    PROMPT_TEMPLATE_VERSION,
    SHORT_CONTEXT_TOKENS,
    WorkloadProfile,
    default_matrix,
    generation_settings_for,
    generate_workloads,
    matrix_from_axes,
)
from .instrumentation import InstrumentationRecorder, utc_now
from .mock_backend import WorkloadMockLLM
from .runner import AgentTaskRunner

BackendFactory = Callable[[], Any]

CLOCK_NOTE = (
    "External EC2 monitoring must use the same run_id and UTC clock. "
    "Cross-machine correlation assumes synchronized clocks; this benchmark "
    "does not verify clock synchronization."
)
CACHE_NOTE = (
    "Local benchmark code cannot reset or verify server cache state. "
    "Warm-up treatment is identical across matched A/B conditions unless the "
    "experiment intentionally studies warming."
)
MEMORY_NOTE = (
    "This local benchmark measures only the Python client process RSS "
    "(peak_benchmark_process_rss_bytes). It cannot measure llama-server RSS, "
    "model weights, GPU VRAM, or KV-cache memory. Those fields are left null "
    "for later merge from independent EC2 telemetry."
)
EXTERNAL_MEMORY_PLACEHOLDERS = {
    "peak_model_server_rss_bytes": None,
    "peak_gpu_vram_mib": None,
    "peak_kv_cache_memory_bytes": None,
}
_ISOLATED_FIXED_FIELDS = ("context_tokens", "max_tokens", "n_steps", "tool_delay_s")
_AXIS_FIELD = {
    "context": "context_tokens",
    "response": "max_tokens",
    "steps": "n_steps",
    "tool_delay": "tool_delay_s",
    "prefix": "prefix_mode",
    "retained_context": "context_mode",
}


@dataclass(frozen=True)
class BenchConfig:
    out_dir: Path
    concurrency_levels: tuple[int, ...] = (1, 2, 4)
    mixed_concurrency: int = 2
    isolated_tasks: int = 8
    mixed_tasks_per_profile: int = 4
    seed: int = 0
    request_timeout_s: float | None = 60.0
    context_tokens: Mapping[str, int] | None = None
    response_tokens: Mapping[str, int] | None = None
    step_counts: Sequence[int] = (1, 3)
    tool_delays_s: Sequence[float] = (0.0, 0.05)
    backend_label: str = "mock"
    run_id: str | None = None
    n_trials: int = 1
    warmup_enabled: bool = False
    warmup_requests: int = 8
    endpoint: str | None = None
    model: str | None = None
    backend_timeout_s: float | None = None
    api_style: str | None = None
    command_line: tuple[str, ...] | None = None
    oversize_context_tokens: int = OVERSIZE_CONTEXT_TOKENS
    oversize_tasks: int = 4


@dataclass(frozen=True)
class SuitePlan:
    name: str
    profiles: tuple[WorkloadProfile, ...]
    concurrency: int
    tasks_per_profile: int
    task_id_prefix: str
    varied_axis: str | None
    measured: bool = True


def quick_config(out_dir: Path, **overrides: Any) -> BenchConfig:
    kwargs: dict[str, Any] = {
        "out_dir": out_dir,
        "concurrency_levels": (1, 2),
        "mixed_concurrency": 2,
        "isolated_tasks": 3,
        "mixed_tasks_per_profile": 1,
        "context_tokens": {"short": 12, "long": 24},
        "response_tokens": {"short": 6, "long": 12},
        "step_counts": (1, 2),
        "tool_delays_s": (0.0, 0.02),
        "request_timeout_s": 5.0,
        "oversize_context_tokens": 48,
        "oversize_tasks": 1,
    }
    kwargs.update(overrides)
    return BenchConfig(**kwargs)


def default_mock_backend() -> WorkloadMockLLM:
    return WorkloadMockLLM(
        delay_s=0.004,
        per_prompt_token_delay_s=0.00012,
        per_token_delay_s=0.0007,
    )


def token_maps(config: BenchConfig) -> tuple[dict[str, int], dict[str, int]]:
    ctx = dict(config.context_tokens or {"short": SHORT_CONTEXT_TOKENS, "long": LONG_CONTEXT_TOKENS})
    resp = dict(config.response_tokens or {"short": 32, "long": 256})
    return ctx, resp


def isolated_tool_delays(config: BenchConfig) -> tuple[float, ...]:
    delays = [float(value) for value in config.tool_delays_s]
    if 0.0 not in delays:
        delays.insert(0, 0.0)
    if not any(value > 0 for value in delays):
        delays.append(0.05)
    return tuple(delays)


def isolated_suite_profiles(config: BenchConfig) -> dict[str, tuple[WorkloadProfile, ...]]:
    ctx, resp = token_maps(config)
    cheap_ctx = {"short": ctx["short"]}
    cheap_resp = {"short": resp["short"]}
    cheap_steps = (min(int(s) for s in config.step_counts),)
    cheap_delay = (0.0,)
    return {
        "isolated_context": matrix_from_axes(ctx, cheap_resp, cheap_steps, cheap_delay),
        "isolated_response": matrix_from_axes(cheap_ctx, resp, cheap_steps, cheap_delay),
        "isolated_steps": matrix_from_axes(
            cheap_ctx, cheap_resp, tuple(int(s) for s in config.step_counts), cheap_delay,
        ),
        "isolated_tool_delay": matrix_from_axes(
            cheap_ctx, cheap_resp, (max(int(s) for s in config.step_counts),),
            isolated_tool_delays(config),
        ),
    }


def prefix_reuse_profiles(config: BenchConfig) -> tuple[WorkloadProfile, ...]:
    ctx, resp = token_maps(config)
    context_class = "long" if "long" in ctx else next(iter(ctx))
    response_class = "short" if "short" in resp else next(iter(resp))
    common = {
        "context_class": context_class,
        "response_class": response_class,
        "n_steps": 1,
        "tool_delay_s": 0.0,
        "context_tokens": ctx[context_class],
        "max_tokens": resp[response_class],
        "context_mode": "fresh",
    }
    return (
        WorkloadProfile(name="prefix_unique_long", prefix_mode="unique", **common),  # type: ignore[arg-type]
        WorkloadProfile(
            name="prefix_shared_long",
            prefix_mode="shared",
            shared_prefix_id="shared-long-v1",
            **common,  # type: ignore[arg-type]
        ),
    )


def retained_context_profiles(config: BenchConfig) -> tuple[WorkloadProfile, ...]:
    ctx, resp = token_maps(config)
    context_class = "long" if "long" in ctx else next(iter(ctx))
    response_class = "short" if "short" in resp else next(iter(resp))
    n_steps = max(2, max(int(s) for s in config.step_counts))
    common = {
        "context_class": context_class,
        "response_class": response_class,
        "n_steps": n_steps,
        "tool_delay_s": 0.0,
        "context_tokens": ctx[context_class],
        "max_tokens": resp[response_class],
        "prefix_mode": "unique",
    }
    return (
        WorkloadProfile(name="context_retained_multistep", context_mode="retained", **common),  # type: ignore[arg-type]
        WorkloadProfile(name="context_fresh_length_matched", context_mode="fresh", **common),  # type: ignore[arg-type]
    )


def oversize_context_profiles(config: BenchConfig) -> tuple[WorkloadProfile, ...]:
    """Prompts sized to miss the llama.cpp slot window; not mixed into other suites."""
    _ctx, resp = token_maps(config)
    response_class = "short" if "short" in resp else next(iter(resp))
    return (
        WorkloadProfile(
            name="oversize_context_admission",
            context_class="long",
            response_class=response_class,  # type: ignore[arg-type]
            n_steps=1,
            tool_delay_s=0.0,
            context_tokens=int(config.oversize_context_tokens),
            max_tokens=int(resp[response_class]),
            prefix_mode="unique",
            context_mode="fresh",
        ),
    )


def axis_values(profiles: Sequence[WorkloadProfile], field: str) -> set[Any]:
    return {getattr(profile, field) for profile in profiles}


def assert_isolated_axis(profiles: Sequence[WorkloadProfile], axis: str) -> None:
    if not profiles:
        raise ValueError("isolated suite produced no profiles")
    varied = _AXIS_FIELD[axis]
    extra_fixed = []
    if axis != "prefix":
        extra_fixed.append("prefix_mode")
    if axis != "retained_context":
        extra_fixed.append("context_mode")
    for field in list(_ISOLATED_FIXED_FIELDS) + extra_fixed:
        values = axis_values(profiles, field)
        if field == varied:
            if len(values) < 2:
                raise ValueError(f"isolated {axis} suite must vary {field}")
        elif len(values) != 1:
            raise ValueError(f"isolated {axis} suite must hold {field} fixed, got {values}")


def plan_suites(config: BenchConfig) -> tuple[SuitePlan, ...]:
    isolated = isolated_suite_profiles(config)
    assert_isolated_axis(isolated["isolated_context"], "context")
    assert_isolated_axis(isolated["isolated_response"], "response")
    assert_isolated_axis(isolated["isolated_steps"], "steps")
    assert_isolated_axis(isolated["isolated_tool_delay"], "tool_delay")
    delays = axis_values(isolated["isolated_tool_delay"], "tool_delay_s")
    if 0.0 not in delays or not any(float(d) > 0 for d in delays):
        raise ValueError("isolated_tool_delay must include a zero-delay control and a nonzero delay")
    ctx, resp = token_maps(config)
    mixed = default_matrix(
        context_tokens=ctx,
        response_tokens=resp,
        step_counts=tuple(int(s) for s in config.step_counts),
        tool_delays_s=tuple(float(d) for d in config.tool_delays_s),
    )
    prefix = prefix_reuse_profiles(config)
    retained = retained_context_profiles(config)
    assert_isolated_axis(prefix, "prefix")
    assert_isolated_axis(retained, "retained_context")
    plans = [
        SuitePlan("isolated_context", isolated["isolated_context"], config.mixed_concurrency,
                  config.isolated_tasks, "iso-ctx", "context"),
        SuitePlan("isolated_response", isolated["isolated_response"], config.mixed_concurrency,
                  config.isolated_tasks, "iso-resp", "response"),
        SuitePlan("isolated_steps", isolated["isolated_steps"], config.mixed_concurrency,
                  config.isolated_tasks, "iso-steps", "steps"),
        SuitePlan("isolated_tool_delay", isolated["isolated_tool_delay"], config.mixed_concurrency,
                  config.isolated_tasks, "iso-tool", "tool_delay"),
        SuitePlan(
            "oversize_context", oversize_context_profiles(config), config.mixed_concurrency,
            max(1, min(int(config.oversize_tasks), int(config.isolated_tasks))),
            "oversize", "oversize_context",
        ),
        SuitePlan("prefix_reuse", prefix, config.mixed_concurrency,
                  config.isolated_tasks, "prefix", "prefix"),
        SuitePlan("retained_context", retained, config.mixed_concurrency,
                  config.isolated_tasks, "retain", "retained_context"),
        SuitePlan("mixed_matrix", mixed, config.mixed_concurrency,
                  config.mixed_tasks_per_profile, "mixed", None),
    ]
    for concurrency in config.concurrency_levels:
        plans.append(SuitePlan(
            f"concurrency_{concurrency}", mixed, concurrency,
            config.mixed_tasks_per_profile, f"conc{concurrency}", None,
        ))
    return tuple(plans)


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if p <= 0:
        return ordered[0]
    if p >= 100:
        return ordered[-1]
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _stats(values: Sequence[float]) -> dict[str, float | None]:
    nums = [float(v) for v in values]
    mean = (sum(nums) / len(nums)) if nums else None
    return {
        "n": len(nums),
        "mean": mean,
        "min": min(nums) if nums else None,
        "p50": percentile(nums, 50),
        "p95": percentile(nums, 95),
        "p99": percentile(nums, 99),
        "max": max(nums) if nums else None,
    }


def current_rss_bytes() -> int:
    statm = Path("/proc/self/statm")
    if statm.exists():
        pages = int(statm.read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(out.strip() or "0") * 1024


def response_length_separation(recorder: InstrumentationRecorder) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for rec in recorder.requests:
        grouped[int(rec.max_tokens)].append(float(rec.token_usage.get("completion_tokens", 0)))
    observed = {str(limit): _stats(values) for limit, values in grouped.items()}
    payload: dict[str, Any] = {
        "applicable": len(grouped) >= 2,
        "configured_max_tokens": sorted(grouped),
        "observed_completion_tokens_by_max_tokens": observed,
        "classes_separated_by_observed_length": None,
        "note": "Suite does not contain multiple configured response-token limits.",
    }
    if len(grouped) < 2:
        return payload
    medians = [(limit, percentile(values, 50) or 0.0) for limit, values in grouped.items()]
    medians.sort()
    short_median, long_median = medians[0][1], medians[-1][1]
    separated = long_median >= short_median + 2 and long_median > short_median * 1.1
    payload["classes_separated_by_observed_length"] = separated
    payload["note"] = (
        "Observed completion lengths differ across configured response classes."
        if separated else
        "Configured short/long max_tokens are not separated by observed completion "
        "length; do not claim response-length effects."
    )
    return payload


def summarize(recorder: InstrumentationRecorder, wall_time_s: float) -> dict[str, Any]:
    request_total = [float(r.timing_metadata.get("total_time_ms") or 0) for r in recorder.requests]
    request_queue = [float(r.timing_metadata.get("queue_time_ms") or 0) for r in recorder.requests]
    request_api = [float(r.timing_metadata.get("api_time_ms") or 0) for r in recorder.requests]
    task_times = [t.task_duration_ms for t in recorder.tasks]
    statuses = Counter(r.status for r in recorder.requests)
    task_statuses = Counter(t.status for t in recorder.tasks)
    n_ok_tasks = sum(1 for t in recorder.tasks if t.status == "success")
    n_failed_req = int(statuses.get("failed", 0))
    n_timeout_req = int(statuses.get("timeout", 0))
    n_cancelled_req = int(statuses.get("cancelled", 0))
    return {
        "wall_time_s": wall_time_s,
        "n_requests": len(recorder.requests),
        "n_tasks": len(recorder.tasks),
        "n_success_tasks": n_ok_tasks,
        "n_failed_requests": n_failed_req,
        "n_timeout_requests": n_timeout_req,
        "n_cancelled_requests": n_cancelled_req,
        "throughput_requests_per_s": (len(recorder.requests) / wall_time_s) if wall_time_s else 0.0,
        "throughput_tasks_per_s": (len(recorder.tasks) / wall_time_s) if wall_time_s else 0.0,
        "success_task_fraction": (n_ok_tasks / len(recorder.tasks)) if recorder.tasks else 0.0,
        "request_status_counts": dict(statuses),
        "task_status_counts": dict(task_statuses),
        "request_total_ms": _stats(request_total),
        "request_queue_ms": _stats(request_queue),
        "request_api_ms": _stats(request_api),
        "task_completion_ms": _stats(task_times),
        "tool_wait_ms": _stats([r.tool_wait_ms for r in recorder.requests]),
        "prompt_tokens": _stats([r.token_usage.get("prompt_tokens", 0) for r in recorder.requests]),
        "completion_tokens": _stats([r.token_usage.get("completion_tokens", 0) for r in recorder.requests]),
        "configured_max_tokens": _stats([r.max_tokens for r in recorder.requests]),
        "observed_completion_tokens": _stats(
            [r.token_usage.get("completion_tokens", 0) for r in recorder.requests]
        ),
        "response_length_separation": response_length_separation(recorder),
        "successful_only": _successful_only_metrics(recorder, wall_time_s),
    }


def _successful_only_metrics(recorder: InstrumentationRecorder, wall_time_s: float) -> dict[str, Any]:
    ok_requests = [r for r in recorder.requests if r.status == "success"]
    ok_tasks = [t for t in recorder.tasks if t.status == "success"]
    return {
        "note": (
            "Latency and token stats over successful requests/tasks only. "
            "Throughput uses suite wall_time_s. Failed and rejected requests "
            "remain in the all-request metrics."
        ),
        "n_requests": len(ok_requests),
        "n_tasks": len(ok_tasks),
        "throughput_requests_per_s": (len(ok_requests) / wall_time_s) if wall_time_s else 0.0,
        "throughput_tasks_per_s": (len(ok_tasks) / wall_time_s) if wall_time_s else 0.0,
        "request_total_ms": _stats(
            [float(r.timing_metadata.get("total_time_ms") or 0) for r in ok_requests]
        ),
        "request_queue_ms": _stats(
            [float(r.timing_metadata.get("queue_time_ms") or 0) for r in ok_requests]
        ),
        "request_api_ms": _stats(
            [float(r.timing_metadata.get("api_time_ms") or 0) for r in ok_requests]
        ),
        "task_completion_ms": _stats([t.task_duration_ms for t in ok_tasks]),
        "prompt_tokens": _stats([r.token_usage.get("prompt_tokens", 0) for r in ok_requests]),
        "completion_tokens": _stats(
            [r.token_usage.get("completion_tokens", 0) for r in ok_requests]
        ),
    }


def resolve_run_id(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def local_package_versions() -> dict[str, str | None]:
    names = ("httpx", "matplotlib")
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return {name: None for name in names}
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _trial_spread(values: Sequence[Any]) -> dict[str, Any]:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return {"median_per_trial": None, "min": None, "max": None, "iqr": None, "n_trials": 0}
    p25 = percentile(nums, 25)
    p75 = percentile(nums, 75)
    return {
        "median_per_trial": percentile(nums, 50),
        "min": min(nums),
        "max": max(nums),
        "iqr": None if p25 is None or p75 is None else p75 - p25,
        "n_trials": len(nums),
    }


def labeled_spread(values: Sequence[Any], label: str) -> dict[str, Any]:
    payload = _trial_spread(values)
    payload["label"] = label
    return payload


def aggregate_suite_metrics(trial_suites: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [suite["metrics"] for suite in trial_suites]
    rss = [suite.get("peak_benchmark_process_rss_bytes", suite.get("peak_rss_bytes")) for suite in trial_suites]
    scalars = (
        "wall_time_s", "n_requests", "n_tasks", "n_success_tasks",
        "n_failed_requests", "n_timeout_requests", "n_cancelled_requests",
        "throughput_requests_per_s", "throughput_tasks_per_s", "success_task_fraction",
    )
    nested = (
        "request_total_ms", "request_queue_ms", "request_api_ms", "task_completion_ms",
        "tool_wait_ms", "prompt_tokens", "completion_tokens", "configured_max_tokens",
        "observed_completion_tokens",
    )
    out: dict[str, Any] = {
        "aggregation": "median_of_per_trial_metrics",
        "n_trials": len(trial_suites),
    }
    for key in scalars:
        out[key] = labeled_spread([m.get(key) for m in metrics], f"median per-trial {key}")
    for key in nested:
        out[key] = {
            stat: labeled_spread(
                [m.get(key, {}).get(stat) for m in metrics],
                f"median per-trial {key} {stat}",
            )
            for stat in ("mean", "min", "p50", "p95", "p99", "max")
        }
    for key in ("request_status_counts", "task_status_counts"):
        statuses: set[str] = set()
        for metric in metrics:
            statuses.update(str(s) for s in metric.get(key, {}))
        out[key] = {
            status: labeled_spread(
                [metric.get(key, {}).get(status, 0) for metric in metrics],
                f"median per-trial {key}.{status}",
            )
            for status in sorted(statuses)
        }
    out["peak_benchmark_process_rss_bytes"] = labeled_spread(
        rss, "median per-trial peak_benchmark_process_rss_bytes",
    )
    success_only = [m.get("successful_only") or {} for m in metrics]
    out["successful_only"] = {
        key: labeled_spread(
            [block.get(key) for block in success_only],
            f"median per-trial successful_only {key}",
        )
        for key in (
            "n_requests", "n_tasks",
            "throughput_requests_per_s", "throughput_tasks_per_s",
        )
    }
    out["successful_only"].update({
        key: {
            stat: labeled_spread(
                [block.get(key, {}).get(stat) for block in success_only],
                f"median per-trial successful_only {key} {stat}",
            )
            for stat in ("mean", "min", "p50", "p95", "p99", "max")
        }
        for key in (
            "request_total_ms", "request_queue_ms", "request_api_ms", "task_completion_ms",
        )
    })
    return out


def suite_summary_row(suite: Mapping[str, Any]) -> dict[str, Any]:
    metrics = suite["metrics"]
    rss = suite.get("peak_benchmark_process_rss_bytes", suite.get("peak_rss_bytes", 0))
    return {
        "name": suite["name"],
        "concurrency": suite["concurrency"],
        "n_requests": metrics["n_requests"],
        "n_tasks": metrics["n_tasks"],
        "throughput_requests_per_s": metrics["throughput_requests_per_s"],
        "throughput_tasks_per_s": metrics["throughput_tasks_per_s"],
        "n_success_tasks": metrics.get("n_success_tasks"),
        "success_task_fraction": metrics["success_task_fraction"],
        "request_status_counts": metrics["request_status_counts"],
        "task_status_counts": metrics["task_status_counts"],
        "n_failed_requests": metrics.get("n_failed_requests"),
        "n_timeout_requests": metrics.get("n_timeout_requests"),
        "request_total_ms": metrics["request_total_ms"],
        "request_queue_ms": metrics["request_queue_ms"],
        "request_api_ms": metrics["request_api_ms"],
        "task_completion_ms": metrics["task_completion_ms"],
        "tool_wait_ms": metrics["tool_wait_ms"],
        "configured_max_tokens": metrics.get("configured_max_tokens"),
        "observed_completion_tokens": metrics.get("observed_completion_tokens"),
        "response_length_separation": metrics.get("response_length_separation"),
        "successful_only": metrics.get("successful_only"),
        "request_total_ms_p95": metrics["request_total_ms"]["p95"],
        "task_completion_ms_p95": metrics["task_completion_ms"]["p95"],
        "peak_rss_mib": rss / (1024 * 1024),
        "peak_benchmark_process_rss_bytes": rss,
        "peak_benchmark_process_rss_mib": rss / (1024 * 1024),
        **EXTERNAL_MEMORY_PLACEHOLDERS,
        "suite_started_at": suite.get("suite_started_at"),
        "suite_finished_at": suite.get("suite_finished_at"),
    }


def intended_workload_rows(
    *,
    suite: str,
    trial_id: int,
    specs: Sequence[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_index, spec in enumerate(specs):
        for step in spec.steps:
            rows.append({
                "suite": suite,
                "trial_id": trial_id,
                "intended_task_launch_index": task_index,
                "profile_name": spec.profile_name,
                "task_id": spec.task_id,
                "request_id": f"{spec.task_id}-step-{step.step_id}",
                "step_id": step.step_id,
                "context_tokens": step.context_tokens,
                "max_tokens": step.max_tokens,
                "n_steps": spec.profile.n_steps,
                "tool_delay_s": step.tool_delay_s,
                "generation_settings": generation_settings_for(step.max_tokens),
                "prefix_mode": spec.profile.prefix_mode,
                "context_mode": spec.profile.context_mode,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "instruction": step.instruction,
                "seed": spec.seed,
            })
    return rows


def merge_observed_workload_rows(
    rows: Sequence[Mapping[str, Any]],
    recorder: InstrumentationRecorder,
) -> list[dict[str, Any]]:
    by_id = {rec.request_id: rec for rec in recorder.requests}
    merged: list[dict[str, Any]] = []
    for row in rows:
        rec = by_id.get(str(row["request_id"]))
        item = dict(row)
        if rec is not None:
            item["prompt_sha256"] = rec.prompt_sha256
            item["submit_launch_seq"] = rec.submit_launch_seq
            item["enqueue_seq"] = rec.enqueue_seq
            item["queue_entry_time"] = rec.queue_entry_time
            item["status"] = rec.status
        merged.append(item)
    return merged


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _write_suite_files(directory: Path, suite: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in suite.items() if k not in {"requests", "tasks"}}
    recorder_dump = {"requests": suite["requests"], "tasks": suite["tasks"]}
    (directory / f"{suite['name']}.metrics.json").write_text(
        json.dumps(slim, indent=2), encoding="utf-8",
    )
    (directory / f"{suite['name']}.trace.json").write_text(
        json.dumps(recorder_dump), encoding="utf-8",
    )


def _suite_dict(
    name: str,
    concurrency: int,
    recorder: InstrumentationRecorder,
    wall_time_s: float,
    peak_rss_bytes: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "concurrency": concurrency,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_benchmark_process_rss_bytes": peak_rss_bytes,
        "peak_benchmark_process_rss_mib": peak_rss_bytes / (1024 * 1024),
        **EXTERNAL_MEMORY_PLACEHOLDERS,
        "metrics": summarize(recorder, wall_time_s),
        "requests": recorder.to_dict()["requests"],
        "tasks": recorder.to_dict()["tasks"],
    }
    if extra:
        payload.update(extra)
    return payload


async def _close_backend(backend: Any) -> None:
    closer = getattr(backend, "aclose", None)
    if not callable(closer):
        return
    result = closer()
    if asyncio.iscoroutine(result):
        await result


async def _sample_rss(stop: asyncio.Event, samples: list[int]) -> None:
    while not stop.is_set():
        try:
            samples.append(current_rss_bytes())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.05)
        except TimeoutError:
            continue


async def _run_suite(
    plan: SuitePlan,
    *,
    seed: int,
    backend_factory: BackendFactory,
    request_timeout_s: float | None,
    run_id: str,
    trial_id: int,
    backend_label: str,
    endpoint: str | None,
    hostname: str,
    pid: int,
    boundary_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    specs = generate_workloads(
        plan.profiles,
        tasks_per_profile=plan.tasks_per_profile,
        seed=seed,
        task_id_prefix=plan.task_id_prefix,
    )
    intended = intended_workload_rows(suite=plan.name, trial_id=trial_id, specs=specs)
    recorder = InstrumentationRecorder(run_id=run_id, trial_id=trial_id, suite=plan.name)
    samples: list[int] = [current_rss_bytes()]
    stop = asyncio.Event()
    sampler = asyncio.create_task(_sample_rss(stop, samples))
    started = time.perf_counter()
    started_at = utc_now()
    _append_jsonl(boundary_path, {
        "event": "suite_started",
        "run_id": run_id,
        "trial_id": trial_id,
        "suite": plan.name,
        "timestamp_utc": started_at,
        "hostname": hostname,
        "pid": pid,
        "backend_label": backend_label,
        "endpoint": endpoint,
        "measured": plan.measured,
    })
    backend = backend_factory()
    try:
        async with FIFOScheduler(
            backend, concurrency=plan.concurrency, request_timeout_s=request_timeout_s,
        ) as scheduler:
            await AgentTaskRunner(scheduler, recorder).run_tasks(specs)
    finally:
        await _close_backend(backend)
    wall = time.perf_counter() - started
    stop.set()
    await sampler
    samples.append(current_rss_bytes())
    recorder.assign_enqueue_seq()
    finished_at = utc_now()
    _append_jsonl(boundary_path, {
        "event": "suite_finished",
        "run_id": run_id,
        "trial_id": trial_id,
        "suite": plan.name,
        "timestamp_utc": finished_at,
        "hostname": hostname,
        "pid": pid,
        "backend_label": backend_label,
        "endpoint": endpoint,
        "measured": plan.measured,
    })
    suite = _suite_dict(plan.name, plan.concurrency, recorder, wall, max(samples), extra={
        "profile_names": [p.name for p in plan.profiles],
        "profiles": [asdict(p) for p in plan.profiles],
        "varied_axis": plan.varied_axis,
        "tasks_per_profile": plan.tasks_per_profile,
        "n_specs": len(specs),
        "suite_started_at": started_at,
        "suite_finished_at": finished_at,
        "finished_at": finished_at,
        "run_id": run_id,
        "trial_id": trial_id,
        "backend_label": backend_label,
        "endpoint": endpoint,
        "hostname": hostname,
        "pid": pid,
        "measured": plan.measured,
        "response_length_separation": response_length_separation(recorder),
    })
    return suite, merge_observed_workload_rows(intended, recorder)


def _warmup_plan(config: BenchConfig) -> SuitePlan:
    ctx, resp = token_maps(config)
    profile = WorkloadProfile(
        name="warmup",
        context_class="short",
        response_class="short",
        n_steps=1,
        tool_delay_s=0.0,
        context_tokens=min(ctx.values()),
        max_tokens=min(resp.values()),
        prefix_mode="unique",
        context_mode="fresh",
    )
    return SuitePlan(
        name="warmup",
        profiles=(profile,),
        concurrency=config.mixed_concurrency,
        tasks_per_profile=max(1, int(config.warmup_requests)),
        task_id_prefix="warmup",
        varied_axis=None,
        measured=False,
    )


def _config_dict(config: BenchConfig, ctx: Mapping[str, int], resp: Mapping[str, int]) -> dict[str, Any]:
    return {
        "concurrency_levels": list(config.concurrency_levels),
        "mixed_concurrency": config.mixed_concurrency,
        "isolated_tasks": config.isolated_tasks,
        "mixed_tasks_per_profile": config.mixed_tasks_per_profile,
        "seed": config.seed,
        "context_tokens": dict(ctx),
        "response_tokens": dict(resp),
        "step_counts": list(config.step_counts),
        "tool_delays_s": list(config.tool_delays_s),
        "backend_label": config.backend_label,
        "request_timeout_s": config.request_timeout_s,
        "n_trials": config.n_trials,
        "warmup_enabled": config.warmup_enabled,
        "warmup_requests": config.warmup_requests,
        "endpoint": config.endpoint,
        "model": config.model,
        "backend_timeout_s": config.backend_timeout_s,
        "api_style": config.api_style,
        "run_id": config.run_id,
        "suite_order": "fixed",
        "task_order": "fixed_generate_workloads_order",
        "oversize_context_tokens": config.oversize_context_tokens,
        "oversize_tasks": config.oversize_tasks,
        "context_budget_note": (
            "Default long context is 512 whitespace tokens so prompts fit one "
            "llama.cpp slot at --ctx-size 4096 --parallel 2. oversize_context "
            "keeps 2048-word pads as an intentional admission/reject suite."
        ),
    }


def build_run_manifest(
    config: BenchConfig,
    *,
    run_id: str,
    hostname: str,
    pid: int,
    created_at: str,
) -> dict[str, Any]:
    ctx, resp = token_maps(config)
    return {
        "run_id": run_id,
        "created_at": created_at,
        "hostname": hostname,
        "pid": pid,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": local_package_versions(),
        "git_revision": git_revision(),
        "command_line": list(config.command_line) if config.command_line is not None else list(sys.argv),
        "benchmark": _config_dict(config, ctx, resp),
        "seed": config.seed,
        "backend": {
            "label": config.backend_label,
            "endpoint": config.endpoint,
            "model": config.model,
            "request_timeout_s": config.request_timeout_s,
            "backend_timeout_s": config.backend_timeout_s,
            "api_style": config.api_style,
        },
        "concurrency_levels": list(config.concurrency_levels),
        "warmup": {
            "enabled": config.warmup_enabled,
            "requests": config.warmup_requests if config.warmup_enabled else 0,
            "policy": "per_trial_before_measured_suites" if config.warmup_enabled else "disabled",
            "server_cache_note": CACHE_NOTE,
        },
        "clock_note": CLOCK_NOTE,
        "memory_note": MEMORY_NOTE,
        **EXTERNAL_MEMORY_PLACEHOLDERS,
    }


async def run_microbenchmarks(
    config: BenchConfig,
    backend_factory: BackendFactory | None = None,
) -> dict[str, Any]:
    if config.n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    factory = backend_factory or default_mock_backend
    config.out_dir.mkdir(parents=True, exist_ok=True)
    ctx, resp = token_maps(config)
    run_id = resolve_run_id(config.run_id)
    hostname = socket.gethostname()
    pid = os.getpid()
    created_at = utc_now()
    boundary_path = config.out_dir / "suite_boundary_events.jsonl"
    if boundary_path.exists():
        boundary_path.unlink()
    plans = plan_suites(config)
    workload_rows: list[dict[str, Any]] = []
    warmup_runs: list[dict[str, Any]] = []
    trial_reports: list[dict[str, Any]] = []
    correlation_windows: list[dict[str, Any]] = []

    for trial_id in range(1, config.n_trials + 1):
        trial_dir = config.out_dir / f"trial_{trial_id}" if config.n_trials > 1 else config.out_dir
        trial_dir.mkdir(parents=True, exist_ok=True)
        if config.warmup_enabled:
            warmup_suite, warmup_rows = await _run_suite(
                _warmup_plan(config),
                seed=config.seed,
                backend_factory=factory,
                request_timeout_s=config.request_timeout_s,
                run_id=run_id,
                trial_id=trial_id,
                backend_label=config.backend_label,
                endpoint=config.endpoint,
                hostname=hostname,
                pid=pid,
                boundary_path=boundary_path,
            )
            warmup_runs.append(warmup_suite)
            workload_rows.extend(warmup_rows)
            (trial_dir / "warmup.metrics.json").write_text(
                json.dumps({k: v for k, v in warmup_suite.items() if k not in {"requests", "tasks"}}, indent=2),
                encoding="utf-8",
            )
            (trial_dir / "warmup.trace.json").write_text(
                json.dumps({"requests": warmup_suite["requests"], "tasks": warmup_suite["tasks"]}),
                encoding="utf-8",
            )
        suites: list[dict[str, Any]] = []
        for plan in plans:
            suite, rows = await _run_suite(
                plan,
                seed=config.seed,
                backend_factory=factory,
                request_timeout_s=config.request_timeout_s,
                run_id=run_id,
                trial_id=trial_id,
                backend_label=config.backend_label,
                endpoint=config.endpoint,
                hostname=hostname,
                pid=pid,
                boundary_path=boundary_path,
            )
            suites.append(suite)
            workload_rows.extend(rows)
            _write_suite_files(trial_dir, suite)
            correlation_windows.append({
                "run_id": run_id,
                "trial_id": trial_id,
                "suite": suite["name"],
                "suite_started_at": suite["suite_started_at"],
                "suite_finished_at": suite["suite_finished_at"],
                "backend_label": config.backend_label,
                "endpoint": config.endpoint,
                "hostname": hostname,
                "pid": pid,
                "peak_benchmark_process_rss_bytes": suite["peak_benchmark_process_rss_bytes"],
                **EXTERNAL_MEMORY_PLACEHOLDERS,
            })
        trial_report = {
            "trial_id": trial_id,
            "seed": config.seed,
            "suite_order": [plan.name for plan in plans],
            "suites": suites,
        }
        (trial_dir / "results.json").write_text(json.dumps(trial_report, indent=2), encoding="utf-8")
        trial_reports.append(trial_report)

    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trial_reports:
        for suite in trial["suites"]:
            by_name[suite["name"]].append(suite)
    aggregates = {
        name: aggregate_suite_metrics(items)
        for name, items in by_name.items()
    }
    primary_suites = trial_reports[0]["suites"]
    report = {
        "created_at": created_at,
        "run_id": run_id,
        "python": sys.version,
        "config": _config_dict(config, ctx, resp),
        "suites": primary_suites,
        "trials": trial_reports,
        "aggregates": aggregates,
        "warmup": warmup_runs,
        "n_trials": config.n_trials,
        "hostname": hostname,
        "pid": pid,
        **EXTERNAL_MEMORY_PLACEHOLDERS,
        "memory_note": MEMORY_NOTE,
        "clock_note": CLOCK_NOTE,
    }
    (config.out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if config.n_trials == 1:
        for suite in primary_suites:
            _write_suite_files(config.out_dir, suite)
    manifest = build_run_manifest(
        config, run_id=run_id, hostname=hostname, pid=pid, created_at=created_at,
    )
    (config.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (config.out_dir / "workload_manifest.json").write_text(
        json.dumps({
            "run_id": run_id,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "intended_fifo_task_launch_order": "generate_workloads order; tasks are launched concurrently with asyncio.gather",
            "enqueue_order_note": (
                "enqueue_seq is ranked by observed scheduler queue_entry_time, "
                "not coroutine launch order."
            ),
            "entries": workload_rows,
        }, indent=2),
        encoding="utf-8",
    )
    (config.out_dir / "telemetry_correlation.json").write_text(
        json.dumps({
            "run_id": run_id,
            "clock_note": CLOCK_NOTE,
            "memory_note": MEMORY_NOTE,
            "windows": correlation_windows,
            **EXTERNAL_MEMORY_PLACEHOLDERS,
        }, indent=2),
        encoding="utf-8",
    )
    return report
