"""Run the Task 3 microbenchmark suites against the existing FIFO scheduler."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scheduler import FIFOScheduler

from .definitions import WorkloadProfile, default_matrix, generate_workloads
from .instrumentation import InstrumentationRecorder, utc_now
from .mock_backend import WorkloadMockLLM
from .runner import AgentTaskRunner

BackendFactory = Callable[[], Any]


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


def quick_config(out_dir: Path) -> BenchConfig:
    return BenchConfig(
        out_dir=out_dir,
        concurrency_levels=(1, 2),
        mixed_concurrency=2,
        isolated_tasks=3,
        mixed_tasks_per_profile=1,
        context_tokens={"short": 12, "long": 24},
        response_tokens={"short": 6, "long": 12},
        step_counts=(1, 2),
        tool_delays_s=(0.0, 0.02),
        request_timeout_s=5.0,
    )


def default_mock_backend() -> WorkloadMockLLM:
    return WorkloadMockLLM(
        delay_s=0.004,
        per_prompt_token_delay_s=0.00012,
        per_token_delay_s=0.0007,
    )


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
    import subprocess
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)
    return int(out.strip() or "0") * 1024


def summarize(recorder: InstrumentationRecorder, wall_time_s: float) -> dict[str, Any]:
    request_total = [float(r.timing_metadata.get("total_time_ms") or 0) for r in recorder.requests]
    request_queue = [float(r.timing_metadata.get("queue_time_ms") or 0) for r in recorder.requests]
    request_api = [float(r.timing_metadata.get("api_time_ms") or 0) for r in recorder.requests]
    task_times = [t.task_duration_ms for t in recorder.tasks]
    statuses = Counter(r.status for r in recorder.requests)
    task_statuses = Counter(t.status for t in recorder.tasks)
    n_ok_tasks = sum(1 for t in recorder.tasks if t.status == "success")
    return {
        "wall_time_s": wall_time_s,
        "n_requests": len(recorder.requests),
        "n_tasks": len(recorder.tasks),
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
    }


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
        "metrics": summarize(recorder, wall_time_s),
        "requests": recorder.to_dict()["requests"],
        "tasks": recorder.to_dict()["tasks"],
    }
    if extra:
        payload.update(extra)
    return payload


async def _sample_rss(stop: asyncio.Event, samples: list[int]) -> None:
    while not stop.is_set():
        samples.append(current_rss_bytes())
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.05)
        except TimeoutError:
            continue


async def _run_suite(
    name: str,
    profiles: Sequence[WorkloadProfile],
    *,
    concurrency: int,
    tasks_per_profile: int,
    seed: int,
    backend_factory: BackendFactory,
    request_timeout_s: float | None,
    task_id_prefix: str,
) -> dict[str, Any]:
    specs = generate_workloads(
        profiles,
        tasks_per_profile=tasks_per_profile,
        seed=seed,
        task_id_prefix=task_id_prefix,
    )
    recorder = InstrumentationRecorder()
    samples: list[int] = [current_rss_bytes()]
    stop = asyncio.Event()
    sampler = asyncio.create_task(_sample_rss(stop, samples))
    started = time.perf_counter()
    async with FIFOScheduler(
        backend_factory(), concurrency=concurrency, request_timeout_s=request_timeout_s,
    ) as scheduler:
        await AgentTaskRunner(scheduler, recorder).run_tasks(specs)
    wall = time.perf_counter() - started
    stop.set()
    await sampler
    samples.append(current_rss_bytes())
    return _suite_dict(name, concurrency, recorder, wall, max(samples), extra={
        "profile_names": [p.name for p in profiles],
        "tasks_per_profile": tasks_per_profile,
        "n_specs": len(specs),
        "finished_at": utc_now(),
    })


def _hold_constant(config: BenchConfig, **overrides: Any) -> tuple[WorkloadProfile, ...]:
    kwargs: dict[str, Any] = {
        "context_tokens": config.context_tokens or {"short": 128, "long": 2048},
        "response_tokens": config.response_tokens or {"short": 32, "long": 256},
        "step_counts": tuple(config.step_counts),
        "tool_delays_s": tuple(config.tool_delays_s),
    }
    kwargs.update(overrides)
    # Isolated sweeps keep the unused axis at the short/cheap setting.
    return default_matrix(**kwargs)


async def run_microbenchmarks(
    config: BenchConfig,
    backend_factory: BackendFactory | None = None,
) -> dict[str, Any]:
    factory = backend_factory or default_mock_backend
    config.out_dir.mkdir(parents=True, exist_ok=True)
    ctx = config.context_tokens or {"short": 128, "long": 2048}
    resp = config.response_tokens or {"short": 32, "long": 256}
    cheap_ctx = {"short": ctx["short"]}
    cheap_resp = {"short": resp["short"]}
    cheap_steps = (min(config.step_counts),)
    cheap_delay = (0.0,)

    suites: list[dict[str, Any]] = []
    suites.append(await _run_suite(
        "isolated_context",
        _hold_constant(
            config, context_tokens=ctx, response_tokens=cheap_resp,
            step_counts=cheap_steps, tool_delays_s=cheap_delay,
        ),
        concurrency=config.mixed_concurrency,
        tasks_per_profile=config.isolated_tasks,
        seed=config.seed,
        backend_factory=factory,
        request_timeout_s=config.request_timeout_s,
        task_id_prefix="iso-ctx",
    ))
    suites.append(await _run_suite(
        "isolated_response",
        _hold_constant(
            config, context_tokens=cheap_ctx, response_tokens=resp,
            step_counts=cheap_steps, tool_delays_s=cheap_delay,
        ),
        concurrency=config.mixed_concurrency,
        tasks_per_profile=config.isolated_tasks,
        seed=config.seed,
        backend_factory=factory,
        request_timeout_s=config.request_timeout_s,
        task_id_prefix="iso-resp",
    ))
    suites.append(await _run_suite(
        "isolated_steps",
        _hold_constant(
            config, context_tokens=cheap_ctx, response_tokens=cheap_resp,
            step_counts=tuple(config.step_counts), tool_delays_s=cheap_delay,
        ),
        concurrency=config.mixed_concurrency,
        tasks_per_profile=config.isolated_tasks,
        seed=config.seed,
        backend_factory=factory,
        request_timeout_s=config.request_timeout_s,
        task_id_prefix="iso-steps",
    ))
    suites.append(await _run_suite(
        "isolated_tool_delay",
        _hold_constant(
            config, context_tokens=cheap_ctx, response_tokens=cheap_resp,
            step_counts=(max(config.step_counts),), tool_delays_s=tuple(config.tool_delays_s),
        ),
        concurrency=config.mixed_concurrency,
        tasks_per_profile=config.isolated_tasks,
        seed=config.seed,
        backend_factory=factory,
        request_timeout_s=config.request_timeout_s,
        task_id_prefix="iso-tool",
    ))
    mixed_profiles = default_matrix(
        context_tokens=ctx,
        response_tokens=resp,
        step_counts=tuple(config.step_counts),
        tool_delays_s=tuple(config.tool_delays_s),
    )
    suites.append(await _run_suite(
        "mixed_matrix",
        mixed_profiles,
        concurrency=config.mixed_concurrency,
        tasks_per_profile=config.mixed_tasks_per_profile,
        seed=config.seed,
        backend_factory=factory,
        request_timeout_s=config.request_timeout_s,
        task_id_prefix="mixed",
    ))
    for concurrency in config.concurrency_levels:
        suites.append(await _run_suite(
            f"concurrency_{concurrency}",
            mixed_profiles,
            concurrency=concurrency,
            tasks_per_profile=config.mixed_tasks_per_profile,
            seed=config.seed,
            backend_factory=factory,
            request_timeout_s=config.request_timeout_s,
            task_id_prefix=f"conc{concurrency}",
        ))

    report = {
        "created_at": utc_now(),
        "python": sys.version,
        "config": {
            "concurrency_levels": list(config.concurrency_levels),
            "mixed_concurrency": config.mixed_concurrency,
            "isolated_tasks": config.isolated_tasks,
            "mixed_tasks_per_profile": config.mixed_tasks_per_profile,
            "seed": config.seed,
            "context_tokens": dict(ctx),
            "response_tokens": dict(resp),
            "step_counts": list(config.step_counts),
            "tool_delays_s": list(config.tool_delays_s),
        },
        "suites": suites,
    }
    (config.out_dir / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for suite in suites:
        slim = {k: v for k, v in suite.items() if k not in {"requests", "tasks"}}
        recorder_dump = {"requests": suite["requests"], "tasks": suite["tasks"]}
        (config.out_dir / f"{suite['name']}.metrics.json").write_text(
            json.dumps(slim, indent=2), encoding="utf-8",
        )
        (config.out_dir / f"{suite['name']}.trace.json").write_text(
            json.dumps(recorder_dump), encoding="utf-8",
        )
    return report
