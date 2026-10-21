import asyncio
import json
import unittest
from datetime import datetime
from unittest.mock import patch

from scheduler import FIFOScheduler, InferenceRequest


def request(index=0):
    return {
        "request_id": f"req-{index}", "task_id": f"agent-{index}",
        "step_id": index, "prompt": f"prompt-{index}",
        "generation_settings": {"max_tokens": 128, "temperature": 0},
    }


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


class ControlledBackend:
    """Event gates make ordering checks independent of machine speed."""

    def __init__(self):
        self.started = []
        self.finished = []
        self.gates = {}
        self.active = 0
        self.peak = 0

    async def generate(self, value):
        rid = value["request_id"]
        gate = self.gates.setdefault(rid, asyncio.Event())
        self.started.append(rid)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await gate.wait()
            self.finished.append(rid)
            return {"response": value["prompt"], "token_usage": {"completion_tokens": 3}}
        finally:
            self.active -= 1

    def release(self, index):
        self.gates.setdefault(f"req-{index}", asyncio.Event()).set()


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def make_scheduler(self, backend, **kwargs):
        scheduler = FIFOScheduler(backend, **kwargs)
        self.addAsyncCleanup(scheduler.shutdown, graceful=False)
        return scheduler

    async def test_fifo_concurrency_and_response_mapping(self):
        for concurrency in (1, 2, 4):
            with self.subTest(concurrency=concurrency):
                backend = ControlledBackend()
                scheduler = await self.make_scheduler(backend, concurrency=concurrency)
                tasks = [asyncio.create_task(scheduler.submit(request(i))) for i in range(8)]
                await eventually(lambda: len(backend.started) == concurrency)
                self.assertEqual(backend.started, [f"req-{i}" for i in range(concurrency)])
                self.assertEqual(scheduler.get_status("req-0"), "running")
                self.assertEqual(scheduler.get_status("req-7"), "queued")
                # Free workers in reverse order: completion need not be FIFO.
                for i in reversed(range(concurrency)):
                    backend.release(i)
                    await tasks[i]
                for i in range(concurrency, 8):
                    backend.release(i)
                results = await asyncio.gather(*tasks)
                self.assertEqual(backend.started, [f"req-{i}" for i in range(8)])
                self.assertEqual(backend.peak, concurrency)
                self.assertEqual(backend.active, 0)
                for i, result in enumerate(results):
                    self.assertEqual(result["request_id"], f"req-{i}")
                    self.assertEqual(result["task_id"], f"agent-{i}")
                    self.assertEqual(result["step_id"], i)
                    self.assertEqual(result["response"], f"prompt-{i}")
                    self.assertEqual(result["token_usage"], {"completion_tokens": 3})
                    self.assertEqual(result["status"], "success")
                    self.assertIsNone(result["error"])
                    self.assertIn(result["worker_id"], range(1, concurrency + 1))
                    self.assertTrue({
                        "request_id", "task_id", "step_id", "response", "status",
                        "timing_metadata", "token_usage",
                    }.issubset(result))
                    timing = result["timing_metadata"]
                    self.assertTrue({
                        "queue_entry_time", "execution_start_time", "execution_finish_time",
                    }.issubset(timing))
                    self.assertNotIn("dispatch_time", timing)
                    self.assertNotIn("finish_time", timing)
                    json.dumps(result)
                self.assertIsNone(scheduler.get_status("req-0"))
                await scheduler.shutdown()

    async def test_timing_uses_monotonic_clock(self):
        backend = ControlledBackend()
        scheduler = await self.make_scheduler(backend)
        # Three clock samples per request: enqueue, dispatch, finish.
        with patch("scheduler.fifo.time.perf_counter", side_effect=[10, 11, 12, 15, 16, 20]):
            first = asyncio.create_task(scheduler.submit(request(0)))
            await eventually(lambda: len(backend.started) == 1)
            second = asyncio.create_task(scheduler.submit(request(1)))
            await eventually(lambda: scheduler.get_status("req-1") == "queued")
            backend.release(0)
            await first
            await eventually(lambda: len(backend.started) == 2)
            backend.release(1)
            result = await second
        timing = result["timing_metadata"]
        self.assertEqual(timing["queue_time_ms"], 4000)
        self.assertEqual(timing["api_time_ms"], 4000)
        self.assertEqual(timing["total_time_ms"], 8000)
        times = [datetime.fromisoformat(timing[key].replace("Z", "+00:00"))
                 for key in ("queue_entry_time", "execution_start_time", "execution_finish_time")]
        self.assertEqual(times, sorted(times))

    async def test_failure_timeout_and_invalid_backend_do_not_lose_worker(self):
        class Backend:
            async def generate(self, value):
                index = value["step_id"]
                if index == 0:
                    raise OSError("server unavailable")
                if index == 1:
                    await asyncio.Event().wait()
                if index == 2:
                    return {"wrong": "shape"}
                if index == 3:
                    return {"response": "ok", "token_usage": {"prompt_tokens": -1}}
                if index == 4:
                    raise asyncio.CancelledError()
                return {"response": "recovered"}

        scheduler = await self.make_scheduler(Backend(), request_timeout_s=0.02)
        results = await asyncio.wait_for(asyncio.gather(*[
            scheduler.submit(request(i)) for i in range(6)
        ]), 2)
        self.assertEqual([r["status"] for r in results],
                         ["failed", "timeout", "failed", "failed", "failed", "success"])
        self.assertIn("server unavailable", results[0]["error"])
        self.assertTrue(all(r["error"] for r in results[:-1]))

    async def test_graceful_shutdown_drains_and_rejects_new_work(self):
        backend = ControlledBackend()
        scheduler = await self.make_scheduler(backend)
        tasks = [asyncio.create_task(scheduler.submit(request(i))) for i in range(3)]
        await eventually(lambda: scheduler.get_status("req-2") == "queued")
        closing = asyncio.create_task(scheduler.shutdown())
        await eventually(lambda: scheduler._closing)
        with self.assertRaises(RuntimeError):
            await scheduler.submit(request(3))
        self.assertFalse(closing.done())
        for i in range(3):
            backend.release(i)
        await asyncio.wait_for(closing, 2)
        self.assertTrue(all(r["status"] == "success" for r in await asyncio.gather(*tasks)))
        self.assertTrue(all(worker.done() for worker in scheduler._workers))
        await scheduler.shutdown()

    async def test_abort_resolves_running_and_queued_requests(self):
        backend = ControlledBackend()
        scheduler = await self.make_scheduler(backend, concurrency=2)
        tasks = [asyncio.create_task(scheduler.submit(request(i))) for i in range(5)]
        await eventually(lambda: len(backend.started) == 2)
        await asyncio.wait_for(scheduler.shutdown(graceful=False), 2)
        results = await asyncio.gather(*tasks)
        self.assertTrue(all(r["status"] == "cancelled" for r in results))
        for result in results[2:]:
            self.assertIsNone(result["worker_id"])
            self.assertIsNone(result["timing_metadata"]["execution_start_time"])
            self.assertEqual(result["timing_metadata"]["api_time_ms"], 0)
        self.assertEqual(backend.active, 0)
        self.assertEqual(scheduler._pending, {})
        await asyncio.wait_for(scheduler._queue.join(), 2)

    async def test_cancelled_submitter_does_not_cancel_accepted_request(self):
        backend = ControlledBackend()
        scheduler = await self.make_scheduler(backend)
        waiter = asyncio.create_task(scheduler.submit(request()))
        await eventually(lambda: bool(backend.started))
        future = scheduler._pending["req-0"].future
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        backend.release(0)
        await scheduler.shutdown()
        self.assertEqual(future.result()["status"], "success")

    async def test_cancelled_shutdown_waiter_does_not_interrupt_cleanup(self):
        backend = ControlledBackend()
        scheduler = await self.make_scheduler(backend)
        work = asyncio.create_task(scheduler.submit(request()))
        await eventually(lambda: bool(backend.started))
        closing = asyncio.create_task(scheduler.shutdown())
        await eventually(lambda: scheduler._closing)
        closing.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await closing
        backend.release(0)
        await asyncio.wait_for(scheduler.shutdown(), 2)
        self.assertEqual((await work)["status"], "success")

    async def test_duplicate_id_and_request_snapshot(self):
        backend = ControlledBackend()
        scheduler = await self.make_scheduler(backend)
        blocker = asyncio.create_task(scheduler.submit(request(0)))
        await eventually(lambda: bool(backend.started))
        value = request(1)
        waiter = asyncio.create_task(scheduler.submit(value))
        await eventually(lambda: scheduler.get_status("req-1") == "queued")
        with self.assertRaises(ValueError):
            await scheduler.submit(value)
        value["prompt"] = "mutated"
        backend.release(0)
        backend.release(1)
        await blocker
        self.assertEqual((await waiter)["response"], "prompt-1")

    async def test_abort_before_any_dispatch(self):
        backend = ControlledBackend()
        scheduler = await self.make_scheduler(backend)
        waiter = asyncio.create_task(scheduler.submit(request()))
        # submit enqueues and creates workers, but they have not yet run.
        await asyncio.sleep(0)
        await scheduler.shutdown(graceful=False)
        self.assertEqual((await waiter)["status"], "cancelled")
        self.assertEqual(backend.started, [])
        await asyncio.wait_for(scheduler._queue.join(), 2)

    async def test_no_timeout_with_immediate_backend(self):
        class Backend:
            async def generate(self, value):
                return {"response": value["prompt"]}

        scheduler = await self.make_scheduler(Backend(), request_timeout_s=None)
        result = await scheduler.submit(InferenceRequest(**request()))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["token_usage"], {})

    async def test_empty_shutdown_and_context_manager(self):
        backend = ControlledBackend()
        scheduler = await self.make_scheduler(backend)
        await scheduler.shutdown()
        with self.assertRaises(RuntimeError):
            await scheduler.start()
        for exceptional in (False, True):
            scheduler = await self.make_scheduler(backend)
            try:
                async with scheduler:
                    if exceptional:
                        raise ValueError("runner failed")
            except ValueError:
                pass
            self.assertTrue(all(w.done() for w in scheduler._workers))

    def test_validation(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                FIFOScheduler(ControlledBackend(), concurrency=value)
        for value in (0, -1, True, float("nan"), float("inf"), "bad"):
            with self.assertRaises(ValueError):
                FIFOScheduler(ControlledBackend(), request_timeout_s=value)
        for override in ({"request_id": ""}, {"task_id": " "}, {"step_id": True},
                         {"step_id": -1}, {"prompt": None}, {"generation_settings": []},
                         {"unknown": 1}):
            with self.assertRaises(ValueError):
                InferenceRequest.parse(request() | override)
        with self.assertRaises(ValueError):
            InferenceRequest.parse({})


if __name__ == "__main__":
    unittest.main()
