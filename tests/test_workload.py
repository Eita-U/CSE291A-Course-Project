import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

from scheduler import FIFOScheduler
from workload import (
    AgentTaskRunner,
    InstrumentationRecorder,
    build_prompt,
    default_matrix,
    generate_task,
    generate_workloads,
)
from workload.definitions import WorkloadProfile, estimate_tokens, prompt_rng
from workload.mock_backend import WorkloadMockLLM


def tiny_matrix():
    return default_matrix(
        context_tokens={"short": 8, "long": 16},
        response_tokens={"short": 4, "long": 8},
        step_counts=(1, 2),
        tool_delays_s=(0.0, 0.02),
    )


class CountingBackend:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.started = []
        self.settings = []
        self.intervals = []

    async def generate(self, request):
        started = time.perf_counter()
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.append(request["request_id"])
        self.settings.append(dict(request["generation_settings"]))
        try:
            await asyncio.sleep(0.01)
            return {
                "response": "ok " + request["prompt"].split()[0],
                "token_usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
        finally:
            self.active -= 1
            self.intervals.append((request["request_id"], started, time.perf_counter()))


class FailingBackend:
    async def generate(self, request):
        if request["step_id"] == 0:
            raise OSError("tool-side model error")
        return {"response": "should-not-run"}


class WorkloadDefinitionTests(unittest.TestCase):
    def test_default_matrix_covers_all_axes(self):
        profiles = default_matrix()
        self.assertEqual(len(profiles), 16)
        names = [profile.name for profile in profiles]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(any(p.context_class == "short" and p.n_steps == 1 for p in profiles))
        self.assertTrue(any(p.context_class == "long" and p.n_steps == 3 for p in profiles))
        self.assertTrue(any(p.response_class == "short" and p.max_tokens == 32 for p in profiles))
        self.assertTrue(any(p.response_class == "long" and p.max_tokens == 256 for p in profiles))
        self.assertTrue(any(p.tool_delay_s == 0.0 for p in profiles))
        self.assertTrue(any(p.tool_delay_s > 0 for p in profiles))

    def test_generation_is_deterministic(self):
        profiles = tiny_matrix()
        first = generate_workloads(profiles, tasks_per_profile=2, seed=7)
        second = generate_workloads(profiles, tasks_per_profile=2, seed=7)
        self.assertEqual(first, second)
        other = generate_workloads(profiles, tasks_per_profile=2, seed=8)
        self.assertNotEqual(first[0].task_id, "")
        self.assertEqual(len(first), len(profiles) * 2)
        prompt_a = build_prompt(
            first[0].steps[0].instruction,
            first[0].steps[0].context_tokens,
            rng=prompt_rng(first[0], first[0].steps[0].step_id),
        )
        prompt_b = build_prompt(
            second[0].steps[0].instruction,
            second[0].steps[0].context_tokens,
            rng=prompt_rng(second[0], second[0].steps[0].step_id),
        )
        self.assertEqual(prompt_a, prompt_b)
        self.assertNotEqual(first, other)

    def test_prompt_matches_token_budget_and_last_step_has_no_tool_delay(self):
        profile = WorkloadProfile(
            name="custom", context_class="short", response_class="long",
            n_steps=3, tool_delay_s=0.05, context_tokens=12, max_tokens=9,
        )
        spec = generate_task(profile, task_index=1, seed=3)
        self.assertEqual([step.step_id for step in spec.steps], [0, 1, 2])
        self.assertEqual([step.tool_delay_s for step in spec.steps], [0.05, 0.05, 0.0])
        self.assertTrue(all(step.max_tokens == 9 for step in spec.steps))
        prompt = build_prompt(spec.steps[0].instruction, 12)
        self.assertEqual(estimate_tokens(prompt), 12)


class WorkloadRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_runner_submits_scheduler_contract_and_records_both_levels(self):
        backend = CountingBackend()
        recorder = InstrumentationRecorder()
        profile = WorkloadProfile(
            name="short_multi", context_class="short", response_class="short",
            n_steps=2, tool_delay_s=0.0, context_tokens=10, max_tokens=4,
        )
        spec = generate_task(profile, seed=1)
        async with FIFOScheduler(backend, concurrency=1, request_timeout_s=2) as scheduler:
            task = await AgentTaskRunner(scheduler, recorder).run_task(spec)
        self.assertEqual(task.status, "success")
        self.assertEqual(task.n_steps_completed, 2)
        self.assertEqual(len(recorder.requests), 2)
        self.assertEqual(backend.settings[0]["max_tokens"], 4)
        self.assertEqual(backend.settings[0]["temperature"], 0)
        first = recorder.requests[0]
        self.assertEqual(first.request_id, f"{spec.task_id}-step-0")
        self.assertEqual(first.task_id, spec.task_id)
        self.assertEqual(first.step_id, 0)
        self.assertIn("queue_entry_time", first.timing_metadata)
        self.assertIn("execution_start_time", first.timing_metadata)
        self.assertIn("execution_finish_time", first.timing_metadata)
        self.assertEqual(first.token_usage["completion_tokens"], 1)
        self.assertEqual(task.completion_tokens, 2)
        json.dumps(recorder.to_dict())

    async def test_tool_wait_does_not_hold_a_scheduler_worker(self):
        backend = CountingBackend()
        profile = WorkloadProfile(
            name="delayed", context_class="short", response_class="short",
            n_steps=2, tool_delay_s=0.05, context_tokens=8, max_tokens=4,
        )
        specs = [
            generate_task(profile, task_index=0, seed=0),
            generate_task(profile, task_index=1, seed=0),
        ]
        async with FIFOScheduler(backend, concurrency=1, request_timeout_s=2) as scheduler:
            tasks = await AgentTaskRunner(scheduler).run_tasks(specs)
        self.assertTrue(all(task.status == "success" for task in tasks))
        self.assertEqual(backend.peak, 1)
        self.assertEqual(len(backend.started), 4)
        first_end = backend.intervals[0][2]
        second_start = backend.intervals[1][1]
        # Tool wait is 50ms and lives outside submit. The next queued request
        # must start well before that wait would have elapsed.
        self.assertLess(second_start - first_end, 0.03)

    async def test_failed_step_stops_the_agent_task(self):
        recorder = InstrumentationRecorder()
        profile = WorkloadProfile(
            name="failing", context_class="short", response_class="short",
            n_steps=3, tool_delay_s=0.01, context_tokens=8, max_tokens=4,
        )
        spec = generate_task(profile)
        async with FIFOScheduler(FailingBackend(), concurrency=1, request_timeout_s=2) as scheduler:
            task = await AgentTaskRunner(scheduler, recorder).run_task(spec)
        self.assertEqual(task.status, "failed")
        self.assertEqual(task.n_steps_completed, 1)
        self.assertEqual(len(recorder.requests), 1)
        self.assertIn("model error", task.error)

    async def test_mock_backend_honors_response_length_and_dump(self):
        recorder = InstrumentationRecorder()
        profile = WorkloadProfile(
            name="long_resp", context_class="short", response_class="long",
            n_steps=1, tool_delay_s=0.0, context_tokens=8, max_tokens=6,
        )
        spec = generate_task(profile)
        async with FIFOScheduler(WorkloadMockLLM(), concurrency=1) as scheduler:
            await AgentTaskRunner(scheduler, recorder).run_task(spec)
        self.assertEqual(recorder.requests[0].token_usage["completion_tokens"], 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            recorder.dump_json(path / "trace.json")
            recorder.dump_jsonl(path / "requests.jsonl", path / "tasks.jsonl")
            payload = json.loads((path / "trace.json").read_text())
            self.assertEqual(len(payload["requests"]), 1)
            self.assertEqual(len(payload["tasks"]), 1)
            self.assertTrue((path / "requests.jsonl").read_text().strip())


if __name__ == "__main__":
    unittest.main()
