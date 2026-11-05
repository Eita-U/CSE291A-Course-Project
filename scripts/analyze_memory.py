#!/usr/bin/env python3
# python3 scripts/analyze_memory.py --run-id phase1-fifo-004
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Analyze EC2 GPU/server memory telemetry by benchmark suite."
)
parser.add_argument(
    "--run-id",
    required=True,
    help="Benchmark run ID, e.g. phase1-fifo-004",
)
parser.add_argument(
    "--results-root",
    type=Path,
    default=Path("results"),
    help="Parent directory containing benchmark result folders.",
)
args = parser.parse_args()

run_id = args.run_id
root = args.results_root / run_id

telemetry_path = root / "gpu_system_metrics.csv"
events_path = root / "suite_boundary_events.jsonl"

REQUIRED_COLUMNS = {
    "utc_time",
    "gpu_memory_used_mib",
    "gpu_memory_total_mib",
    "gpu_utilization_pct",
    "server_rss_bytes",
}

# Present in the updated monitor CSV. Older files omit them; empty cells are
# treated as missing (the llama.cpp metrics drop out after the server exits).
OPTIONAL_FLOAT_COLUMNS = (
    "server_gpu_memory_mib",
    "system_available_bytes",
    "system_used_bytes",
    "prompt_tokens_total",
    "prompt_tokens_cached_total",
    "kv_cache_tokens",
    "kv_cache_tokens_max",
    "kv_cache_usage_ratio",
    "prompt_cache_entries",
    "prompt_cache_tokens",
    "prompt_cache_bytes",
    "prompt_cache_bytes_max",
    "prompt_cache_saves_total",
    "prompt_cache_loads_total",
    "prompt_cache_loaded_tokens_total",
    "prompt_cache_evictions_total",
    "requests_processing",
    "requests_deferred",
)

def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def parse_optional_float(row: dict, key: str):
    if key not in row:
        return None
    raw = row.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    return float(raw)


def peak(values):
    present = [v for v in values if v is not None]
    return max(present) if present else None


def trough(values):
    present = [v for v in values if v is not None]
    return min(present) if present else None


def bytes_to_mib(value):
    if value is None:
        return None
    return round(value / (1024 * 1024), 3)


def counter_delta(selected, field):
    present = [s[field] for s in selected if s.get(field) is not None]
    if len(present) < 2:
        return None
    return present[-1] - present[0]


def median_or_none(values, ndigits=None):
    present = [v for v in values if v is not None]
    if not present:
        return None
    value = statistics.median(present)
    if ndigits is None:
        return value
    return round(value, ndigits)


# ---- Read remote EC2 monitor samples ----
with telemetry_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
    if missing:
        raise SystemExit(f"Missing telemetry columns: {sorted(missing)}")
    present_optional = [
        col for col in OPTIONAL_FLOAT_COLUMNS if col in (reader.fieldnames or [])
    ]

    samples = []
    for row in reader:
        try:
            sample = {
                "time": parse_time(row["utc_time"]),
                "gpu_mib": float(row["gpu_memory_used_mib"]),
                "gpu_total_mib": float(row["gpu_memory_total_mib"]),
                "gpu_util_pct": float(row["gpu_utilization_pct"]),
                "server_rss_bytes": int(float(row["server_rss_bytes"])),
            }
        except (ValueError, KeyError) as exc:
            print(f"Skipping malformed telemetry row: {exc}", file=sys.stderr)
            continue
        for col in present_optional:
            sample[col] = parse_optional_float(row, col)
        samples.append(sample)

if not samples:
    raise SystemExit("No valid monitor samples found.")

samples.sort(key=lambda s: s["time"])
has_llama_metrics = "kv_cache_tokens" in present_optional
has_server_gpu = "server_gpu_memory_mib" in present_optional

def load_suite_windows(events, allowed_run_ids):
    windows = defaultdict(dict)
    seen_run_ids = set()
    for event in events:
        recorded = event.get("run_id")
        if recorded:
            seen_run_ids.add(recorded)
        if allowed_run_ids is not None and recorded not in (None, *allowed_run_ids):
            continue
        event_name = event.get("event")
        trial_id = event.get("trial_id")
        suite = event.get("suite")
        timestamp = event.get("timestamp_utc")
        if trial_id is None or not suite or not timestamp:
            continue
        key = (trial_id, suite)
        if event_name == "suite_started":
            windows[key]["start"] = parse_time(timestamp)
        elif event_name == "suite_finished":
            windows[key]["finish"] = parse_time(timestamp)
    complete = {
        key: window for key, window in windows.items()
        if "start" in window and "finish" in window
    }
    return complete, seen_run_ids


# ---- Read local suite start/end events ----
# The folder is the source of truth. Events may still carry a different
# recorded run_id if --out and --run-id were not updated together.
raw_events = []
with events_path.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            raw_events.append(json.loads(line))

complete_windows, event_run_ids = load_suite_windows(raw_events, (run_id,))
if not complete_windows:
    complete_windows, event_run_ids = load_suite_windows(raw_events, None)
    if complete_windows:
        print(
            "Warning: suite_boundary_events.jsonl has run_id "
            f"{sorted(event_run_ids)}, not {run_id}. "
            "Using every event in this folder.",
            file=sys.stderr,
        )
if not complete_windows:
    raise SystemExit(
        "No complete suite windows found. Inspect suite_boundary_events.jsonl "
        "for event names/timestamp fields."
    )

# Initial post-load / pre-benchmark baseline.
# This assumes monitoring was started before the benchmark, as intended.
baseline_n = min(10, len(samples))
baseline_samples = samples[:baseline_n]
baseline_gpu_mib = statistics.median(s["gpu_mib"] for s in baseline_samples)
baseline_rss_bytes = statistics.median(
    s["server_rss_bytes"] for s in baseline_samples
)
baseline_server_gpu_mib = median_or_none(
    s.get("server_gpu_memory_mib") for s in baseline_samples
)

# Monitor regularity.
gaps_s = [
    (samples[i]["time"] - samples[i - 1]["time"]).total_seconds()
    for i in range(1, len(samples))
]
max_gap_s = max(gaps_s) if gaps_s else 0.0

# ---- Per-trial, per-suite peaks ----
per_suite = []

for (trial_id, suite), window in sorted(
    complete_windows.items(),
    key=lambda item: (str(item[0][0]), item[1]["start"], item[0][1]),
):
    start, finish = window["start"], window["finish"]
    selected = [s for s in samples if start <= s["time"] <= finish]
    llama_selected = [s for s in selected if s.get("kv_cache_tokens") is not None]

    record = {
        "run_id": run_id,
        "trial_id": trial_id,
        "suite": suite,
        "suite_started_at": start.isoformat(),
        "suite_finished_at": finish.isoformat(),
        "duration_s": round((finish - start).total_seconds(), 3),
        "telemetry_sample_count": len(selected),
        "llama_metrics_sample_count": len(llama_selected),
        "monitor_covers_suite_window": (
            samples[0]["time"] <= start and samples[-1]["time"] >= finish
        ),
        "peak_gpu_vram_mib": None,
        "gpu_vram_delta_vs_initial_mib": None,
        "peak_server_gpu_memory_mib": None,
        "server_gpu_memory_delta_vs_initial_mib": None,
        "peak_gpu_utilization_pct": None,
        "peak_model_server_rss_mib": None,
        "server_rss_delta_vs_initial_mib": None,
        "peak_system_used_mib": None,
        "min_system_available_mib": None,
        "peak_kv_cache_tokens": None,
        "peak_kv_cache_usage_ratio": None,
        "kv_cache_tokens_max": None,
        "peak_prompt_cache_tokens": None,
        "peak_prompt_cache_bytes": None,
        "peak_prompt_cache_mib": None,
        "peak_prompt_cache_entries": None,
        "peak_requests_processing": None,
        "peak_requests_deferred": None,
        "prompt_tokens_delta": None,
        "prompt_tokens_cached_delta": None,
        "prompt_cache_hit_fraction": None,
        "prompt_cache_saves_delta": None,
        "prompt_cache_loads_delta": None,
        "prompt_cache_loaded_tokens_delta": None,
        "prompt_cache_evictions_delta": None,
    }

    if selected:
        peak_gpu = max(s["gpu_mib"] for s in selected)
        peak_util = max(s["gpu_util_pct"] for s in selected)
        peak_rss = max(s["server_rss_bytes"] for s in selected)
        peak_server_gpu = peak(s.get("server_gpu_memory_mib") for s in selected)

        record.update({
            "peak_gpu_vram_mib": peak_gpu,
            "gpu_vram_delta_vs_initial_mib": round(
                peak_gpu - baseline_gpu_mib, 3
            ),
            "peak_gpu_utilization_pct": peak_util,
            "peak_model_server_rss_mib": bytes_to_mib(peak_rss),
            "server_rss_delta_vs_initial_mib": round(
                (peak_rss - baseline_rss_bytes) / (1024 * 1024), 3
            ),
            "peak_server_gpu_memory_mib": peak_server_gpu,
            "server_gpu_memory_delta_vs_initial_mib": (
                None
                if peak_server_gpu is None or baseline_server_gpu_mib is None
                else round(peak_server_gpu - baseline_server_gpu_mib, 3)
            ),
            "peak_system_used_mib": bytes_to_mib(
                peak(s.get("system_used_bytes") for s in selected)
            ),
            "min_system_available_mib": bytes_to_mib(
                trough(s.get("system_available_bytes") for s in selected)
            ),
        })

    if llama_selected:
        prompt_delta = counter_delta(llama_selected, "prompt_tokens_total")
        cached_delta = counter_delta(llama_selected, "prompt_tokens_cached_total")
        hit_fraction = None
        if prompt_delta not in (None, 0) and cached_delta is not None:
            hit_fraction = round(cached_delta / prompt_delta, 6)

        peak_prompt_cache_bytes = peak(
            s.get("prompt_cache_bytes") for s in llama_selected
        )
        record.update({
            "peak_kv_cache_tokens": peak(
                s.get("kv_cache_tokens") for s in llama_selected
            ),
            "peak_kv_cache_usage_ratio": peak(
                s.get("kv_cache_usage_ratio") for s in llama_selected
            ),
            "kv_cache_tokens_max": peak(
                s.get("kv_cache_tokens_max") for s in llama_selected
            ),
            "peak_prompt_cache_tokens": peak(
                s.get("prompt_cache_tokens") for s in llama_selected
            ),
            "peak_prompt_cache_bytes": peak_prompt_cache_bytes,
            "peak_prompt_cache_mib": bytes_to_mib(peak_prompt_cache_bytes),
            "peak_prompt_cache_entries": peak(
                s.get("prompt_cache_entries") for s in llama_selected
            ),
            "peak_requests_processing": peak(
                s.get("requests_processing") for s in llama_selected
            ),
            "peak_requests_deferred": peak(
                s.get("requests_deferred") for s in llama_selected
            ),
            "prompt_tokens_delta": prompt_delta,
            "prompt_tokens_cached_delta": cached_delta,
            "prompt_cache_hit_fraction": hit_fraction,
            "prompt_cache_saves_delta": counter_delta(
                llama_selected, "prompt_cache_saves_total"
            ),
            "prompt_cache_loads_delta": counter_delta(
                llama_selected, "prompt_cache_loads_total"
            ),
            "prompt_cache_loaded_tokens_delta": counter_delta(
                llama_selected, "prompt_cache_loaded_tokens_total"
            ),
            "prompt_cache_evictions_delta": counter_delta(
                llama_selected, "prompt_cache_evictions_total"
            ),
        })

    per_suite.append(record)

# ---- Aggregate memory metrics across the three trials ----
by_suite = defaultdict(list)
for record in per_suite:
    if record["telemetry_sample_count"] > 0:
        by_suite[record["suite"]].append(record)

aggregates = []
for suite, records in sorted(by_suite.items()):
    def values(field):
        return [float(r[field]) for r in records if r[field] is not None]

    gpu = values("peak_gpu_vram_mib")
    gpu_delta = values("gpu_vram_delta_vs_initial_mib")
    rss = values("peak_model_server_rss_mib")
    rss_delta = values("server_rss_delta_vs_initial_mib")
    util = values("peak_gpu_utilization_pct")
    server_gpu = values("peak_server_gpu_memory_mib")
    kv_tokens = values("peak_kv_cache_tokens")
    kv_ratio = values("peak_kv_cache_usage_ratio")
    cache_tokens = values("peak_prompt_cache_tokens")
    cache_mib = values("peak_prompt_cache_mib")
    hit = values("prompt_cache_hit_fraction")
    req_proc = values("peak_requests_processing")
    req_def = values("peak_requests_deferred")

    aggregates.append({
        "run_id": run_id,
        "suite": suite,
        "trials_with_telemetry": len(records),
        "all_trial_windows_covered": all(
            r["monitor_covers_suite_window"] for r in records
        ),
        "median_peak_gpu_vram_mib": median_or_none(gpu, 3),
        "min_peak_gpu_vram_mib": min(gpu) if gpu else None,
        "max_peak_gpu_vram_mib": max(gpu) if gpu else None,
        "median_gpu_delta_vs_initial_mib": median_or_none(gpu_delta, 3),
        "median_peak_server_gpu_memory_mib": median_or_none(server_gpu, 3),
        "median_peak_server_rss_mib": median_or_none(rss, 3),
        "min_peak_server_rss_mib": min(rss) if rss else None,
        "max_peak_server_rss_mib": max(rss) if rss else None,
        "median_server_rss_delta_vs_initial_mib": median_or_none(rss_delta, 3),
        "median_peak_gpu_utilization_pct": median_or_none(util, 3),
        "median_peak_kv_cache_tokens": median_or_none(kv_tokens, 3),
        "median_peak_kv_cache_usage_ratio": median_or_none(kv_ratio, 6),
        "median_peak_prompt_cache_tokens": median_or_none(cache_tokens, 3),
        "median_peak_prompt_cache_mib": median_or_none(cache_mib, 3),
        "median_prompt_cache_hit_fraction": median_or_none(hit, 6),
        "median_peak_requests_processing": median_or_none(req_proc, 3),
        "median_peak_requests_deferred": median_or_none(req_def, 3),
    })

first_suite = min(w["start"] for w in complete_windows.values())
last_suite = max(w["finish"] for w in complete_windows.values())
llama_samples = [s for s in samples if s.get("kv_cache_tokens") is not None]

limitations = [
    "gpu_memory_used_mib is GPU-wide usage, not memory attributable to one request.",
    "server_rss_bytes is llama-server CPU resident memory, not GPU VRAM.",
    "The monitor samples approximately once per second; brief spikes can be missed.",
    "Cross-machine suite correlation assumes local and EC2 UTC clocks are sufficiently synchronized.",
]
if has_server_gpu:
    limitations.append(
        "server_gpu_memory_mib is llama-server process GPU memory, still not per-request."
    )
if has_llama_metrics:
    limitations.extend([
        "kv_cache_* and prompt_cache_* come from llama.cpp slots/metrics and may be empty after the server exits.",
        "prompt_tokens_* and prompt_cache_*_total are cumulative counters; per-suite values are end-minus-start in the window.",
        "prompt_cache_hit_fraction is cached_prompt_token_delta / prompt_token_delta in the suite window, not a server-reported hit rate.",
    ])
else:
    limitations.append(
        "This telemetry file does not include KV-cache occupancy or prefix-cache columns."
    )

overall = {
    "run_id": run_id,
    "monitor": {
        "sample_count": len(samples),
        "llama_metrics_sample_count": len(llama_samples),
        "optional_columns_present": present_optional,
        "first_sample_utc": samples[0]["time"].isoformat(),
        "last_sample_utc": samples[-1]["time"].isoformat(),
        "maximum_observed_sample_gap_s": round(max_gap_s, 3),
    },
    "benchmark": {
        "first_suite_started_utc": first_suite.isoformat(),
        "last_suite_finished_utc": last_suite.isoformat(),
        "full_measured_suite_range_covered": (
            samples[0]["time"] <= first_suite
            and samples[-1]["time"] >= last_suite
        ),
    },
    "initial_post_load_baseline": {
        "method": f"median of first {baseline_n} monitor samples",
        "gpu_vram_mib": baseline_gpu_mib,
        "model_server_rss_mib": bytes_to_mib(baseline_rss_bytes),
        "server_gpu_memory_mib": baseline_server_gpu_mib,
    },
    "overall_peaks": {
        "peak_gpu_vram_mib": max(s["gpu_mib"] for s in samples),
        "peak_gpu_utilization_pct": max(s["gpu_util_pct"] for s in samples),
        "peak_model_server_rss_mib": bytes_to_mib(
            max(s["server_rss_bytes"] for s in samples)
        ),
        "peak_server_gpu_memory_mib": peak(
            s.get("server_gpu_memory_mib") for s in samples
        ),
        "peak_kv_cache_tokens": peak(
            s.get("kv_cache_tokens") for s in llama_samples
        ),
        "peak_kv_cache_usage_ratio": peak(
            s.get("kv_cache_usage_ratio") for s in llama_samples
        ),
        "kv_cache_tokens_max": peak(
            s.get("kv_cache_tokens_max") for s in llama_samples
        ),
        "peak_prompt_cache_tokens": peak(
            s.get("prompt_cache_tokens") for s in llama_samples
        ),
        "peak_prompt_cache_mib": bytes_to_mib(
            peak(s.get("prompt_cache_bytes") for s in llama_samples)
        ),
        "peak_prompt_cache_entries": peak(
            s.get("prompt_cache_entries") for s in llama_samples
        ),
        "peak_requests_processing": peak(
            s.get("requests_processing") for s in llama_samples
        ),
        "peak_requests_deferred": peak(
            s.get("requests_deferred") for s in llama_samples
        ),
        "prompt_tokens_total_end": (
            llama_samples[-1]["prompt_tokens_total"] if llama_samples else None
        ),
        "prompt_tokens_cached_total_end": (
            llama_samples[-1]["prompt_tokens_cached_total"]
            if llama_samples
            else None
        ),
    },
    "limitations": limitations,
}

# ---- Write outputs ----
(root / "memory_analysis.json").write_text(
    json.dumps(overall, indent=2), encoding="utf-8"
)
(root / "per_suite_memory.json").write_text(
    json.dumps(per_suite, indent=2), encoding="utf-8"
)
(root / "per_suite_memory_aggregate.json").write_text(
    json.dumps(aggregates, indent=2), encoding="utf-8"
)


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


write_csv(root / "per_suite_memory.csv", per_suite)
write_csv(root / "per_suite_memory_aggregate.csv", aggregates)

# ---- Human-readable output ----
print("\n=== Monitor validation ===")
print(f"Samples: {len(samples)}")
print(f"Llama.cpp metric samples: {len(llama_samples)}")
print(f"Monitor range: {samples[0]['time'].isoformat()} -> {samples[-1]['time'].isoformat()}")
print(f"Maximum sample gap: {max_gap_s:.2f} s")
print(f"Benchmark range: {first_suite.isoformat()} -> {last_suite.isoformat()}")
print("Full benchmark coverage:", overall["benchmark"]["full_measured_suite_range_covered"])

print("\n=== Overall memory results ===")
print(f"Initial GPU VRAM baseline: {baseline_gpu_mib:.1f} MiB")
print(f"Peak GPU VRAM: {overall['overall_peaks']['peak_gpu_vram_mib']:.1f} MiB")
if baseline_server_gpu_mib is not None:
    print(f"Initial llama-server GPU memory: {baseline_server_gpu_mib:.1f} MiB")
if overall["overall_peaks"]["peak_server_gpu_memory_mib"] is not None:
    print(
        "Peak llama-server GPU memory: "
        f"{overall['overall_peaks']['peak_server_gpu_memory_mib']:.1f} MiB"
    )
print(f"Initial llama-server RSS: {baseline_rss_bytes / 1024**2:.1f} MiB")
print(f"Peak llama-server RSS: {overall['overall_peaks']['peak_model_server_rss_mib']:.1f} MiB")
if overall["overall_peaks"]["peak_kv_cache_tokens"] is not None:
    print(
        "Peak KV cache tokens: "
        f"{overall['overall_peaks']['peak_kv_cache_tokens']:.0f} / "
        f"{overall['overall_peaks']['kv_cache_tokens_max']:.0f} "
        f"(ratio {overall['overall_peaks']['peak_kv_cache_usage_ratio']:.3f})"
    )
if overall["overall_peaks"]["peak_prompt_cache_tokens"] is not None:
    print(
        "Peak prompt cache: "
        f"{overall['overall_peaks']['peak_prompt_cache_tokens']:.0f} tokens, "
        f"{overall['overall_peaks']['peak_prompt_cache_mib']:.1f} MiB"
    )

print("\n=== Per-suite median peaks across trials ===")
for item in aggregates:
    extras = []
    if item["median_peak_kv_cache_tokens"] is not None:
        extras.append(f"median KV={item['median_peak_kv_cache_tokens']} tok")
    if item["median_prompt_cache_hit_fraction"] is not None:
        extras.append(f"median cache-hit={item['median_prompt_cache_hit_fraction']}")
    extra = (", " + ", ".join(extras)) if extras else ""
    print(
        f"{item['suite']}: trials={item['trials_with_telemetry']}, "
        f"covered={item['all_trial_windows_covered']}, "
        f"median GPU={item['median_peak_gpu_vram_mib']} MiB, "
        f"median server RSS={item['median_peak_server_rss_mib']} MiB"
        f"{extra}"
    )

print("\nWrote:")
for name in (
    "memory_analysis.json",
    "per_suite_memory.json",
    "per_suite_memory.csv",
    "per_suite_memory_aggregate.json",
    "per_suite_memory_aggregate.csv",
):
    print(f"  {root / name}")
