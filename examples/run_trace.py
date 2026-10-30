import asyncio
import random
from pathlib import Path
import json

from scheduler import FIFOScheduler
from scheduler.qwen_backend import QwenHTTPBackend
from measurement.trace_loader import load_trace, group_tasks
from measurement.replay_trace import replay_trace


async def main():
    df = load_trace("data/processed/trace_32k.parquet")
    tasks = group_tasks(df)

    random.seed(41)
    random_task_ids = random.sample(list(tasks), 3)
    random_tasks = {task_id: tasks[task_id] for task_id in random_task_ids}
    random_tasks = {
        task_id: tasks[task_id].head(5)
        for task_id in random_task_ids
    }
    

    async with QwenHTTPBackend(
        endpoint="http://127.0.0.1:8080/v1/chat/completions",
        api_style="openai",
        timeout_s=180,
    ) as backend:
        async with FIFOScheduler(
            backend,
            concurrency=1,
            request_timeout_s=180,
        ) as scheduler:
            results = await replay_trace(scheduler, random_tasks)

    output = Path("measurement/results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())