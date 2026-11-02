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


def _suite_or_none(report: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    try:
        return _suite(report, name)
    except KeyError:
        return None


def _peak_rss_mib(suite: Mapping[str, Any]) -> float:
    raw = suite.get("peak_benchmark_process_rss_bytes")
    if raw is None:
        raw = suite.get("peak_rss_bytes") or 0
    if isinstance(raw, dict):
        raw = raw.get("median_per_trial") or 0
    return float(raw) / (1024 * 1024)


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
    sep = (iso_resp.get("metrics") or {}).get("response_length_separation") or {}
    resp_title = "Isolated response max_tokens (configured)"
    if sep.get("classes_separated_by_observed_length") is False:
        resp_title += " — observed lengths not separated"
    fig, ax = plt.subplots()
    bar(ax, list(resp_tot), [sum(v) / len(v) for v in resp_tot.values()],
        "Mean request time (ms)", resp_title, "#3b6ea5")
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

    oversize = _suite_or_none(report, "oversize_context")
    if oversize is not None:
        fig, ax = plt.subplots()
        counts = Counter(r["status"] for r in oversize["requests"])
        bar(ax, list(counts), [counts[k] for k in counts], "Requests",
            "Oversize-context admission (intentional reject class; not mixed into other suites)",
            "#8b4c70")
        save(fig, "oversize_context")

    conc = _concurrency_suites(report)
    xs = [s["concurrency"] for s in conc]
    fig, ax = plt.subplots()
    ax.plot(xs, [s["metrics"]["throughput_requests_per_s"] for s in conc], marker="o", label="all requests/s")
    ax.plot(xs, [s["metrics"]["throughput_tasks_per_s"] for s in conc], marker="s", label="all tasks/s")
    ok_req = [
        (s.get("metrics") or {}).get("successful_only", {}).get("throughput_requests_per_s")
        for s in conc
    ]
    if all(v is not None for v in ok_req):
        ax.plot(xs, ok_req, marker="^", label="successful requests/s")
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
    ax.plot(xs, [_peak_rss_mib(s) for s in conc], marker="o")
    ax.set_xlabel("Scheduler concurrency")
    ax.set_ylabel("Peak benchmark-process RSS (MiB)")
    ax.set_title("Peak benchmark-process RSS vs concurrency (not model/GPU/server memory)")
    save(fig, "concurrency_peak_rss")

    for name, title in (
        ("prefix_reuse", "Matched unique vs shared long prefix (timing only; not a cache-hit claim)"),
        ("retained_context", "Matched retained vs fresh context (length-matched; not a KV-reuse claim)"),
    ):
        extra = _suite_or_none(report, name)
        if extra is None:
            continue
        tot = _request_metric(extra["requests"], "total_time_ms")
        fig, ax = plt.subplots()
        bar(ax, list(tot), [sum(v) / len(v) for v in tot.values()],
            "Mean request time (ms)", title, "#5b4c8b")
        save(fig, name)

    fig, ax = plt.subplots()
    ax.plot(xs, [s["metrics"]["task_completion_ms"]["p95"] for s in conc], marker="o")
    ax.set_xlabel("Scheduler concurrency")
    ax.set_ylabel("Task completion p95 (ms)")
    ax.set_title("Task completion tail vs concurrency")
    save(fig, "concurrency_task_p95")

    aggregates = report.get("aggregates") or {}
    n_trials = int(report.get("n_trials") or 1)
    if n_trials > 1 and aggregates:
        conc_pairs = [(s["name"], s["concurrency"]) for s in conc if s["name"] in aggregates]
        if conc_pairs:

            def _bundle(name: str, *path: str) -> Mapping[str, Any]:
                node: Any = aggregates[name]
                for key in path:
                    node = node[key]
                return node

            def _med(bundle: Mapping[str, Any]) -> float:
                return float(bundle.get("median_per_trial") or 0)

            def _err(bundle: Mapping[str, Any]) -> tuple[float, float]:
                median = _med(bundle)
                low = float(bundle["min"]) if bundle.get("min") is not None else median
                high = float(bundle["max"]) if bundle.get("max") is not None else median
                return median - low, high - median

            xs_a = [concurrency for _, concurrency in conc_pairs]
            fig, ax = plt.subplots()
            tput = [_bundle(name, "throughput_requests_per_s") for name, _ in conc_pairs]
            err = list(zip(*[_err(b) for b in tput]))
            ax.errorbar(xs_a, [_med(b) for b in tput], yerr=err, marker="o", capsize=4,
                        label="median per-trial requests/s")
            ax.set_xlabel("Scheduler concurrency")
            ax.set_ylabel("Throughput")
            ax.set_title("Median per-trial throughput vs concurrency (min–max)")
            ax.legend()
            save(fig, "concurrency_throughput_across_trials")

            fig, ax = plt.subplots()
            p95 = [_bundle(name, "request_total_ms", "p95") for name, _ in conc_pairs]
            err = list(zip(*[_err(b) for b in p95]))
            ax.errorbar(xs_a, [_med(b) for b in p95], yerr=err, marker="s", capsize=4,
                        label="median per-trial request p95")
            ax.set_xlabel("Scheduler concurrency")
            ax.set_ylabel("Request total time (ms)")
            ax.set_title("Median per-trial request p95 vs concurrency (not pooled p95)")
            ax.legend()
            save(fig, "concurrency_tail_latency_across_trials")

            fig, ax = plt.subplots()
            rss_b = [_bundle(name, "peak_benchmark_process_rss_bytes") for name, _ in conc_pairs]
            scale = 1024 * 1024
            err = list(zip(*[(lo / scale, hi / scale) for lo, hi in (_err(b) for b in rss_b)]))
            ax.errorbar(xs_a, [_med(b) / scale for b in rss_b], yerr=err, marker="o", capsize=4)
            ax.set_xlabel("Scheduler concurrency")
            ax.set_ylabel("Peak benchmark-process RSS (MiB)")
            ax.set_title("Median per-trial benchmark-process RSS (not GPU/server)")
            save(fig, "concurrency_peak_rss_across_trials")

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
