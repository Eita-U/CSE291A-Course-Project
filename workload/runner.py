"""Module B: dummy agent workflows over the FIFO scheduler contract.

Each step submits one LLM request. Tool waits run after submit returns and
occupy no scheduler worker. Response length is requested via
generation_settings["max_tokens"] from the workload step spec.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol, Sequence

from .definitions import AgentTaskSpec, build_prompt, estimate_tokens, prompt_rng
from .instrumentation import InstrumentationRecorder, TaskRecord, utc_now


class SchedulerClient(Protocol):
    async def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        ...


class AgentTaskRunner:
    def __init__(
        self,
        scheduler: SchedulerClient,
        recorder: InstrumentationRecorder | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.recorder = recorder or InstrumentationRecorder()

    async def run_task(self, spec: AgentTaskSpec) -> TaskRecord:
        started = time.perf_counter()
        start_utc = utc_now()
        prior_text = ""
        recorded = []
        for step in spec.steps:
            prompt = build_prompt(
                step.instruction,
                step.context_tokens,
                extra_text=prior_text,
                rng=prompt_rng(spec, step.step_id),
            )
            result = await self.scheduler.submit({
                "request_id": f"{spec.task_id}-step-{step.step_id}",
                "task_id": spec.task_id,
                "step_id": step.step_id,
                "prompt": prompt,
                "generation_settings": {
                    "max_tokens": step.max_tokens,
                    "temperature": 0,
                },
            })
            tool_wait_ms = 0.0
            if result.get("status") == "success" and step.tool_delay_s > 0:
                delay_started = time.perf_counter()
                await asyncio.sleep(step.tool_delay_s)
                tool_wait_ms = (time.perf_counter() - delay_started) * 1000
            record = self.recorder.record_request(
                profile_name=spec.profile_name,
                scheduler_response=result,
                prompt_token_estimate=estimate_tokens(prompt),
                max_tokens=step.max_tokens,
                tool_delay_s=step.tool_delay_s,
                tool_wait_ms=tool_wait_ms,
            )
            recorded.append(record)
            if result.get("status") != "success":
                break
            prior_text = str(result.get("response") or "")
        return self.recorder.record_task(
            spec_task_id=spec.task_id,
            profile_name=spec.profile_name,
            n_steps_planned=len(spec.steps),
            request_records=recorded,
            task_start_time=start_utc,
            task_finish_time=utc_now(),
            task_duration_ms=(time.perf_counter() - started) * 1000,
        )

    async def run_tasks(self, specs: Sequence[AgentTaskSpec]) -> list[TaskRecord]:
        """Run independent agent tasks concurrently against one scheduler."""
        return list(await asyncio.gather(*[self.run_task(spec) for spec in specs]))
