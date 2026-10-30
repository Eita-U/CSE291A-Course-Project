import asyncio
import pandas as pd
import time
from datetime import datetime, timezone
from tqdm.auto import tqdm


async def replay_task(scheduler, task: pd.DataFrame, progress, counts) -> dict:
    results = []
    task = task.sort_values("step")

    start_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    for row in task.to_dict("records"):
        gap = float(row["pre_gap"])
        tqdm.write(f'{row["session_id"]} step={row["step"]}: pre_gap={gap:.3f}s')

        await asyncio.sleep(float(row["pre_gap"]))
        result = await scheduler.submit({
            "request_id": f'{row["session_id"]}-step-{row["step"]}',
            "task_id": str(row["session_id"]),
            "step_id": int(row["step"]),
            "messages": row["input"],
            "generation_settings": {
                "max_tokens": int(row["output_length"]),
                "temperature": 0,
            },
        })
        results.append(result)

        status = result["status"]
        if status == "failed":
            counts["failed"] += 1
        elif status == "timeout":
            counts["timeout"] += 1

        advance = 1 

        if status != "success":
            skipped = len(task) - len(results)
            counts["skipped"] += skipped
            advance += skipped

        progress.set_postfix(counts, refresh=False)
        progress.update(advance)

        if result["status"] != "success":
            break

    duration_ms = (time.perf_counter() - started) * 1000
    finish_utc = datetime.now(timezone.utc).isoformat()

    return {
        "task_id": str(task.iloc[0]["session_id"]),
        "status": results[-1]["status"] if results else "failed",
        "n_steps_completed": sum(result["status"] == "success" for result in results),
        "timing_metadata": {
            "task_start_time": start_utc,
            "task_finish_time": finish_utc,
            "task_duration_ms": duration_ms,
        },
        "requests": results,
    }


async def replay_trace(scheduler, tasks: dict[str, pd.DataFrame]) -> dict[str, dict]:
    task_ids = list(tasks)

    counts = {"failed": 0, "timeout": 0, "skipped": 0}

    with tqdm(total=sum(len(task) for task in tasks.values()), desc="Replaying requests", unit="req") as progress:
        results = await asyncio.gather(*(replay_task(scheduler, tasks[task_id], progress, counts) for task_id in task_ids))

    return dict(zip(task_ids, results))