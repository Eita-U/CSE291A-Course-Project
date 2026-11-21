from pathlib import Path
import json
import csv
import numpy as np
from datetime import datetime
from collections import defaultdict

def load_data(result_path: str | Path, memory_path: str | Path):
    with open(result_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    with open(memory_path, "r", newline="", encoding="utf-8") as f:
        memory = list(csv.DictReader(f))

    return results, memory


def filter_memory_by_time(results, memory):
    timing = results["timing_metadata"]
    start = datetime.fromisoformat(timing["run_start_time"])
    finish = datetime.fromisoformat(timing["run_finish_time"])

    return [
        row
        for row in memory
        if start <= datetime.fromisoformat(row["utc_time"]) <= finish
    ]

def calculate_output_length_difference(results):
    pairs = [(request["target_output_tokens"], request["token_usage"]["completion_tokens"]) for task in results["tasks"].values() for request in task["requests"] if request["status"] == "success" and request["token_usage"].get("completion_tokens") is not None]

    target_total = sum(target for target, actual in pairs)
    actual_total = sum(actual for target, actual in pairs)

    return {
        "total_difference_pct": (actual_total - target_total) / target_total * 100 if target_total > 0 else None,
        "shorter_requests": {
            "count": sum(actual < target for target, actual in pairs),
            "total": len(pairs),
        },
    }

def calculate_request_memory(results, memory):
    requests = []
    for task in results["tasks"].values():
        for request in task["requests"]:
            timing = request["timing_metadata"]
            if timing["execution_start_time"] is None:
                continue

            start = datetime.fromisoformat(timing["execution_start_time"])
            finish = datetime.fromisoformat(timing["execution_finish_time"])
            
            samples = [row for row in memory if start <= datetime.fromisoformat(row["utc_time"]) <= finish]

            requests.append({
                "request_id": request["request_id"],
                "task_id": request["task_id"],
                "status": request["status"],
                "step_id": int(request["step_id"]),
                "prompt_tokens": request["token_usage"].get("prompt_tokens"),
                "completion_tokens": request["token_usage"].get("completion_tokens"),
                "peak_gpu_memory_mib": max((float(row["gpu_memory_used_mib"]) for row in samples), default=None),
                "peak_server_rss_bytes": max((int(row["server_rss_bytes"]) for row in samples), default=None),
                "api_time_ms": timing["api_time_ms"],
                "sample_count": len(samples)
            })

    groups = defaultdict(list)
    token_bin_size = 4096
    for request in requests:
        if (request["status"] != "success" or request["prompt_tokens"] is None or request["sample_count"] == 0):
            continue

        group_id = request["prompt_tokens"] // token_bin_size
        groups[group_id].append(request)


    result = {}
    for group_id in sorted(groups):
        peak_gpu_memories_mib = [request["peak_gpu_memory_mib"] for request in groups[group_id]]
        peak_server_rss_bytes = [request["peak_server_rss_bytes"] for request in groups[group_id]]
        api_times_ms = [request["api_time_ms"] for request in groups[group_id]]
        completion_tokens = [request["completion_tokens"] for request in groups[group_id] if request["completion_tokens"] is not None]

        gpu_p50,  gpu_p95,  gpu_p99 = np.percentile(peak_gpu_memories_mib, [50, 95, 99], method="linear")
        server_rss_p50,  server_rss_p95,  server_rss_p99 = np.percentile(peak_server_rss_bytes, [50, 95, 99], method="linear")
        api_p50, api_p95, api_p99 = np.percentile(api_times_ms, [50, 95, 99], method="linear")


        result[f"{group_id * token_bin_size} - {(group_id + 1) * token_bin_size}"] = {
            "gpu_mean_mib": sum(peak_gpu_memories_mib) / len(peak_gpu_memories_mib),
            "gpu_p50_mib": float(gpu_p50),
            "gpu_p95_mib": float(gpu_p95),
            "gpu_p99_mib": float(gpu_p99),
            "server_rss_mean_bytes": sum(peak_server_rss_bytes) / len(peak_server_rss_bytes),
            "server_rss_p50_bytes": float(server_rss_p50),
            "server_rss_p95_bytes": float(server_rss_p95),
            "server_rss_p99_bytes": float(server_rss_p99),
            "api_mean_ms": sum(api_times_ms) / len(api_times_ms),
            "api_p50_ms": float(api_p50),
            "api_p95_ms": float(api_p95),
            "api_p99_ms": float(api_p99),
            "mean_completion_tokens": (sum(completion_tokens) / len(completion_tokens) if completion_tokens else None),
            "count": len(groups[group_id]),
        } 

    return result

def calculate_context_growth(results, memory):
    samples = [(datetime.fromisoformat(row["utc_time"]), row) for row in memory]
    records = []

    for task in results["tasks"].values():
        for request in task["requests"]:
            timing = request["timing_metadata"]
            if timing["execution_start_time"] is None:
                continue

            start = datetime.fromisoformat(timing["execution_start_time"])
            finish = datetime.fromisoformat(timing["execution_finish_time"])

            rows = [row for timestamp, row in samples if start <= timestamp <= finish]
            usage = request["token_usage"]
            backend = request.get("backend_timings") or {}

            records.append({
                "task_id": request["task_id"],
                "step_id": request["step_id"],
                "status": request["status"],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "cache_n": backend.get("cache_n"),
                "prompt_n": backend.get("prompt_n"),
                "prompt_ms": backend.get("prompt_ms"),
                "api_time_ms": timing["api_time_ms"],
                "peak_gpu_memory_mib": max((float(row["gpu_memory_used_mib"]) for row in rows), default=None),
                "peak_server_rss_bytes": max((int(row["server_rss_bytes"]) for row in rows), default=None),
                "peak_kv_cache_tokens": max((int(row["kv_cache_tokens"]) for row in rows if row.get("kv_cache_tokens") not in (None, "")), default=None),
                "peak_kv_cache_usage_ratio": max((float(row["kv_cache_usage_ratio"]) for row in rows if row.get("kv_cache_usage_ratio") not in (None, "")), default=None),
                "peak_prompt_cache_bytes": max((int(row["prompt_cache_bytes"]) for row in rows if row.get("prompt_cache_bytes") not in (None, "")), default=None),
                "sample_count": len(rows),
            })

    return records

def calculate_task_completion_time(results):
    completion_times = [task["timing_metadata"]["task_duration_ms"] for task in results["tasks"].values() if task["status"] == "success"]
    p50, p95, p99 = np.percentile(completion_times, [50, 95, 99], method="linear")
    return {
        "count": len(completion_times),
        "min_ms": min(completion_times),
        "mean_ms": sum(completion_times) / len(completion_times),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": max(completion_times),
    }

def calculate_queueing_time(results):
    queueing_times = [request["timing_metadata"]["queue_time_ms"] for task in results["tasks"].values() for request in task["requests"] if request["timing_metadata"]["execution_start_time"] is not None]
    p50, p95, p99 = np.percentile(queueing_times, [50, 95, 99], method="linear")
    return {
        "count": len(queueing_times),
        "min_ms": min(queueing_times),
        "mean_ms": sum(queueing_times) / len(queueing_times),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": max(queueing_times),
    }

def calculate_throughput(results):
    successful_tasks = [task for task in results["tasks"].values() if task["status"] == "success"]
    duration_sec = results["timing_metadata"]["run_duration_ms"] / 1000
    return len(successful_tasks) / duration_sec

def calculate_failures(results):
    failure_tasks = [task for task in results["tasks"].values() if task["status"] == "failed"]
    timeout_tasks = [task for task in results["tasks"].values() if task["status"] == "timeout"]
    return {
        "failure_count": len(failure_tasks),
        "failure_ratio": len(failure_tasks) / len(results["tasks"]),
        "timeout_count": len(timeout_tasks),
        "timeout_ratio": len(timeout_tasks) / len(results["tasks"]),
    }

def calculate_peak_memory(memory):
    total_gpu_memory = float(memory[0]["gpu_memory_total_mib"])
    peak_gpu_memory = max(float(row["gpu_memory_used_mib"]) for row in memory)
    peak_server_rss_memory = max(int(row["server_rss_bytes"]) for row in memory)

    return {
        "peak_gpu_memory": peak_gpu_memory,
        "peak_gpu_memory_pct": peak_gpu_memory / total_gpu_memory * 100,
        "total_gpu_memory": total_gpu_memory,
        "peak_server_rss_bytes": peak_server_rss_memory,
    }

def calculate_prefix_metrics(results):
    timings = [request["backend_timings"] for task in results["tasks"].values() for request in task["requests"] if request.get("backend_timings", {}) != {}]

    return {
        "measured_request_count": len(timings),
        "total_cached_tokens": sum(t["cache_n"] for t in timings),
        "total_processed_tokens": sum(t["prompt_n"] for t in timings),
        "total_prompt_ms": sum(t["prompt_ms"] for t in timings),
    }