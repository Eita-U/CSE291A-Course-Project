import asyncio
import random
from pathlib import Path
import json
import time
from datetime import datetime, timezone

from scheduler import FIFOScheduler
from scheduler.qwen_backend import QwenHTTPBackend
from measurement.trace_loader import load_trace, group_tasks
from measurement.replay_trace import replay_trace


async def main():
    cache_prompt = True
    pre_gap_scale = 1.0
    concurrency = 1
    
    df = load_trace("data/processed/trace_32k.parquet")
    tasks = group_tasks(df)

    random.seed(42)

    # For cache/context growth experiments
    # selected_task_ids = random.sample(list(tasks), 3)
    # task_index = 0
    # task_id = selected_task_ids[task_index]
    # random_tasks = {task_id: tasks[task_id]}

    # For none-cache/context growth experiments
    random_task_ids = random.sample(list(tasks), 1)
    random_tasks = {task_id: tasks[task_id] for task_id in random_task_ids}

    async with QwenHTTPBackend(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        api_style="openai",
        timeout_s=180,
    ) as backend:
        async with FIFOScheduler(
            backend,
            concurrency=concurrency,
            request_timeout_s=180,
        ) as scheduler:
            start_utc = datetime.now(timezone.utc).isoformat()
            started = time.perf_counter()
            
            results = await replay_trace(scheduler, random_tasks, cache_prompt, pre_gap_scale)

            duration_ms = (time.perf_counter() - started) * 1000
            finish_utc = datetime.now(timezone.utc).isoformat()

    results = {
        "config": {
            "concurrency": concurrency,
            "cache_prompt": cache_prompt,
            "pre_gap_scale": pre_gap_scale,
        },
        "timing_metadata": {
            "run_start_time": start_utc,
            "run_finish_time": finish_utc,
            "run_duration_ms": duration_ms,
        },
        "tasks": results
    }
    run_id = (f"concurrency-{concurrency}-gap-{pre_gap_scale}-cache-{cache_prompt}")
    output = Path(f"measurement/results/{run_id}/results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())