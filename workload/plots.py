"""Render microbenchmark graphs from a results.json report."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .benchmarks import percentile


def _suite(report: Mapping[str, Any], name: str) -> dict[str, Any]:
    for suite in report["suites"]:
        if suite["name"] == name:
            return suite
    raise KeyError(name)


def _concurrency_suites(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    suites = [s for s in report["suites"] if str(s["name"]).startswith("concurrency_")]
    return sorted(suites, key=lambda s: int(s["concurrency"]))


def _request_metric(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["profile_name"])].append(float(row["timing_metadata"][key]))
    return dict(grouped)


def _task_metric(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["profile_name"])].append(float(row[key]))
    return dict(grouped)


def _pxx(grouped: Mapping[str, Sequence[float]], p: float) -> list[float]:
    return [percentile(list(grouped[k]), p) or 0.0 for k in grouped]


def _short(name: str) -> str:
    return (
        name.replace("_ctx_", "/")
        .replace("_resp_", "/")
        .replace("step_", "")
        .replace("notool", "-")
        .replace("tool", "T")
    )


def generate_plots(report: Mapping[str, Any], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.figsize": (10, 5.2),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
    })
    written: list[Path] = []

    def save(fig, stem: str) -> None:
        path = out_dir / f"{stem}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)

    def bar(ax, labels, values, ylabel, title, color="#3b6ea5") -> None:
        ax.bar(range(len(labels)), values, color=color)
        ax.set_xticks(range(len(labels)), [_short(x) for x in labels], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    mixed = _suite(report, "mixed_matrix")
    reqs = mixed["requests"]
    tasks = mixed["tasks"]

    totals = _request_metric(reqs, "total_time_ms")
    fig, ax = plt.subplots()
    labels = list(totals)
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], _pxx(totals, 50), 0.4, label="p50")
    ax.bar([i + 0.2 for i in x], _pxx(totals, 95), 0.4, label="p95")
    ax.set_xticks(list(x), [_short(v) for v in labels], rotation=35, ha="right")
    ax.set_ylabel("Request total time (ms)")
    ax.set_title("Mixed matrix request latency")
    ax.legend()
    save(fig, "mixed_request_latency")

    queues = _request_metric(reqs, "queue_time_ms")
    fig, ax = plt.subplots()
    bar(ax, list(queues), _pxx(queues, 95), "Queue time p95 (ms)", "Mixed matrix queueing tail latency", "#c47b2b")
    save(fig, "mixed_queue_p95")

    completions = _task_metric(tasks, "task_duration_ms")
    fig, ax = plt.subplots()
    bar(ax, list(completions), _pxx(completions, 95), "Task completion p95 (ms)",
        "Mixed matrix task completion time", "#2a7f62")
    save(fig, "mixed_task_completion")

    api = _request_metric(reqs, "api_time_ms")
    tool = _task_metric(reqs, "tool_wait_ms")
    fig, ax = plt.subplots()
    labels = list(totals)
    idx = range(len(labels))
    q_mean = [sum(queues[k]) / len(queues[k]) for k in labels]
    a_mean = [sum(api[k]) / len(api[k]) for k in labels]
    t_mean = [sum(tool[k]) / len(tool[k]) for k in labels]
    ax.bar(idx, q_mean, label="queue")
    ax.bar(idx, a_mean, bottom=q_mean, label="inference")
    ax.bar(idx, t_mean, bottom=[q + a for q, a in zip(q_mean, a_mean)], label="tool wait")
    ax.set_xticks(list(idx), [_short(v) for v in labels], rotation=35, ha="right")
    ax.set_ylabel("Mean time (ms)")
    ax.set_title("Mixed matrix latency breakdown")
    ax.legend()
    save(fig, "mixed_latency_breakdown")

    fig, ax = plt.subplots()
    counts = Counter(r["status"] for r in reqs)
    bar(ax, list(counts), [counts[k] for k in counts], "Requests", "Request outcomes", "#8b4c70")
    save(fig, "mixed_request_outcomes")

    fig, ax = plt.subplots()
    prompt = _task_metric(reqs, "prompt_token_estimate")
    completion = {
        name: [float(r["token_usage"].get("completion_tokens", 0)) for r in reqs if r["profile_name"] == name]
        for name in totals
    }
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], [sum(prompt[k]) / len(prompt[k]) for k in labels], 0.4, label="prompt est.")
    ax.bar([i + 0.2 for i in x], [sum(completion[k]) / len(completion[k]) for k in labels], 0.4, label="completion")
    ax.set_xticks(list(x), [_short(v) for v in labels], rotation=35, ha="right")
    ax.set_ylabel("Tokens")
    ax.set_title("Mixed matrix token counts")
    ax.legend()
    save(fig, "mixed_token_counts")

    fig, ax = plt.subplots()
    for name, values in totals.items():
        ordered = sorted(values)
        ys = [(i + 1) / len(ordered) for i in range(len(ordered))]
        ax.plot(ordered, ys, label=_short(name), linewidth=1.2)
    ax.set_xlabel("Request total time (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Mixed matrix request latency CDF")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "mixed_request_cdf")

    iso_ctx = _suite(report, "isolated_context")
    ctx_tot = _request_metric(iso_ctx["requests"], "total_time_ms")
    fig, ax = plt.subplots()
    bar(ax, list(ctx_tot), [sum(v) / len(v) for v in ctx_tot.values()],
        "Mean request time (ms)", "Isolated context length", "#3b6ea5")
    save(fig, "isolated_context")

    iso_resp = _suite(report, "isolated_response")
    resp_tot = _request_metric(iso_resp["requests"], "total_time_ms")
    fig, ax = plt.subplots()
    bar(ax, list(resp_tot), [sum(v) / len(v) for v in resp_tot.values()],
        "Mean request time (ms)", "Isolated response length", "#3b6ea5")
    save(fig, "isolated_response")

    iso_steps = _suite(report, "isolated_steps")
    step_tot = _task_metric(iso_steps["tasks"], "task_duration_ms")
    fig, ax = plt.subplots()
    bar(ax, list(step_tot), [sum(v) / len(v) for v in step_tot.values()],
        "Mean task completion (ms)", "Isolated step count", "#2a7f62")
    save(fig, "isolated_steps")

    iso_tool = _suite(report, "isolated_tool_delay")
    tool_groups = _task_metric(iso_tool["tasks"], "task_duration_ms")
    tool_wait = _task_metric(iso_tool["tasks"], "tool_wait_ms_total")
    fig, ax = plt.subplots()
    labels = list(tool_groups)
    ax.bar(range(len(labels)), [sum(tool_groups[k]) / len(tool_groups[k]) for k in labels], 0.4,
           label="task time")
    ax.bar([i + 0.4 for i in range(len(labels))], [sum(tool_wait[k]) / len(tool_wait[k]) for k in labels],
           0.4, label="tool wait")
    ax.set_xticks([i + 0.2 for i in range(len(labels))], [_short(v) for v in labels], rotation=35, ha="right")
    ax.set_ylabel("Milliseconds")
    ax.set_title("Isolated tool delay")
    ax.legend()
    save(fig, "isolated_tool_delay")

    conc = _concurrency_suites(report)
    xs = [s["concurrency"] for s in conc]
    fig, ax = plt.subplots()
    ax.plot(xs, [s["metrics"]["throughput_requests_per_s"] for s in conc], marker="o", label="requests/s")
    ax.plot(xs, [s["metrics"]["throughput_tasks_per_s"] for s in conc], marker="s", label="tasks/s")
    ax.set_xlabel("Scheduler concurrency")
    ax.set_ylabel("Throughput")
    ax.set_title("Throughput vs concurrency")
    ax.legend()
    save(fig, "concurrency_throughput")

    fig, ax = plt.subplots()
    ax.plot(xs, [s["metrics"]["request_total_ms"]["p50"] for s in conc], marker="o", label="p50")
    ax.plot(xs, [s["metrics"]["request_total_ms"]["p95"] for s in conc], marker="s", label="p95")
    ax.plot(xs, [s["metrics"]["request_total_ms"]["p99"] for s in conc], marker="^", label="p99")
    ax.set_xlabel("Scheduler concurrency")
    ax.set_ylabel("Request total time (ms)")
    ax.set_title("Request tail latency vs concurrency")
    ax.legend()
    save(fig, "concurrency_tail_latency")

    fig, ax = plt.subplots()
    ax.plot(xs, [s["metrics"]["request_queue_ms"]["p50"] for s in conc], marker="o", label="p50")
    ax.plot(xs, [s["metrics"]["request_queue_ms"]["p95"] for s in conc], marker="s", label="p95")
    ax.set_xlabel("Scheduler concurrency")
    ax.set_ylabel("Queue time (ms)")
    ax.set_title("Queueing time vs concurrency")
    ax.legend()
    save(fig, "concurrency_queue_time")

    fig, ax = plt.subplots()
    ax.plot(xs, [s["peak_rss_bytes"] / (1024 * 1024) for s in conc], marker="o")
    ax.set_xlabel("Scheduler concurrency")
    ax.set_ylabel("Peak process RSS (MiB)")
    ax.set_title("Peak process memory vs concurrency (not GPU)")
    save(fig, "concurrency_peak_rss")

    fig, ax = plt.subplots()
    ax.plot(xs, [s["metrics"]["task_completion_ms"]["p95"] for s in conc], marker="o")
    ax.set_xlabel("Scheduler concurrency")
    ax.set_ylabel("Task completion p95 (ms)")
    ax.set_title("Task completion tail vs concurrency")
    save(fig, "concurrency_task_p95")

    index = out_dir / "index.html"
    cards = "\n".join(
        f'<figure><img src="{path.name}" alt="{path.stem}"><figcaption>{path.stem}</figcaption></figure>'
        for path in written
    )
    index.write_text(
        "<!doctype html><meta charset='utf-8'><title>Microbenchmarks</title>"
        "<style>body{font-family:sans-serif;max-width:960px;margin:24px auto;padding:0 16px}"
        "img{width:100%;height:auto}figure{margin:0 0 28px}</style>"
        "<h1>FIFO microbenchmarks</h1>" + cards,
        encoding="utf-8",
    )
    written.append(index)
    return written
