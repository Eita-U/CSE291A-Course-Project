"""Run from repository root: python3 -m examples.workload_run."""

import asyncio
import json

from scheduler import FIFOScheduler
from workload import AgentTaskRunner, InstrumentationRecorder, default_matrix, generate_workloads
from workload.mock_backend import WorkloadMockLLM


async def main() -> None:
    profiles = default_matrix(
        context_tokens={"short": 16, "long": 32},
        response_tokens={"short": 8, "long": 16},
        step_counts=(1, 2),
        tool_delays_s=(0.0, 0.01),
    )
    specs = generate_workloads(profiles, tasks_per_profile=1, seed=0)
    recorder = InstrumentationRecorder()
    async with FIFOScheduler(WorkloadMockLLM(), concurrency=2) as scheduler:
        tasks = await AgentTaskRunner(scheduler, recorder).run_tasks(specs)
    summary = {
        "n_profiles": len(profiles),
        "n_tasks": len(tasks),
        "n_requests": len(recorder.requests),
        "task_statuses": sorted({task.status for task in tasks}),
        "sample_task": recorder.to_dict()["tasks"][0],
        "sample_request": recorder.to_dict()["requests"][0],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
