import json
from pathlib import Path

from measurement.calculate_metrics import load_data, filter_memory_by_time, calculate_failures, calculate_peak_memory, calculate_queueing_time, calculate_task_completion_time, calculate_throughput, calculate_output_length_difference, calculate_prefix_metrics, calculate_context_growth

def main():
    run_id = "concurrency-1-gap-1.0-cache-True"
    results, memory = load_data(f"measurement/results/{run_id}/results.json", "measurement/memory.csv")
    memory = filter_memory_by_time(results, memory)

    metrics = {
        "task_completion_time": calculate_task_completion_time(results),
        "queueing_time": calculate_queueing_time(results),
        "throughput_tasks_per_s": calculate_throughput(results),
        "failures": calculate_failures(results),
        "memory": calculate_peak_memory(memory),
        "calculate_output_length_difference": calculate_output_length_difference(results),
        "prefix_metrics": calculate_prefix_metrics(results),
        "context_growth": calculate_context_growth(results, memory)
    }
    output = Path("measurement/metrics.json")
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
