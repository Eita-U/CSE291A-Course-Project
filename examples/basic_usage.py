"""Run from repository root: python3 -m examples.basic_usage."""

import asyncio
import json

from scheduler import FIFOScheduler
from scheduler.mock_llm import MockLLMBackend


async def agent_task(scheduler: FIFOScheduler, task_id: str) -> list[dict]:
    results = []
    context = "Summarize the observation."
    for step_id in range(2):
        result = await scheduler.submit({
            "request_id": f"{task_id}-step-{step_id}",
            "task_id": task_id,
            "step_id": step_id,
            "prompt": context,
            "generation_settings": {"max_tokens": 128, "temperature": 0},
        })
        results.append(result)
        if result["status"] != "success":
            break
        context += "\n" + result["response"]
        if step_id == 0:
            # Tool waits belong to the runner and hold no inference worker.
            await asyncio.sleep(0.02)
    return results


async def main() -> None:
    async with FIFOScheduler(MockLLMBackend(), concurrency=2) as scheduler:
        results = await asyncio.gather(
            agent_task(scheduler, "agent-1"), agent_task(scheduler, "agent-2"),
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
