import asyncio
import unittest

from scheduler import SlotAffinityScheduler


def request(agent, step):
    return {
        "request_id": f"{agent}-{step}", "task_id": agent, "step_id": step,
        "prompt": f"{agent} step {step}",
        "generation_settings": {"max_tokens": 16, "temperature": 0},
    }


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0)


class SlotBackend:
    """Records the slot each request was pinned to; gates control completion."""

    def __init__(self):
        self.started = []  # (request_id, id_slot)
        self.gates = {}
        self.active = 0
        self.peak = 0

    async def generate(self, value):
        rid = value["request_id"]
        self.started.append((rid, value["generation_settings"]["id_slot"]))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await self.gates.setdefault(rid, asyncio.Event()).wait()
            return {"response": rid, "token_usage": {"completion_tokens": 1}}
        finally:
            self.active -= 1

    def release(self, rid):
        self.gates.setdefault(rid, asyncio.Event()).set()

    def started_ids(self):
        return [rid for rid, _ in self.started]


class SlotAffinityTests(unittest.IsolatedAsyncioTestCase):
    async def make(self, backend, **kwargs):
        scheduler = SlotAffinityScheduler(backend, **kwargs)
        self.addAsyncCleanup(scheduler.shutdown, graceful=False)
        return scheduler

    async def run_step(self, scheduler, backend, agent, step):
        task = asyncio.create_task(scheduler.submit(request(agent, step)))
        await eventually(lambda: f"{agent}-{step}" in backend.started_ids())
        backend.release(f"{agent}-{step}")
        return await task

    async def test_held_slot_goes_back_to_its_agent(self):
        backend = SlotBackend()
        scheduler = await self.make(backend, slots=1, hold_s=10, max_skips=3)
        first = await self.run_step(scheduler, backend, "A", 1)
        self.assertFalse(first["slot_hit"])
        # B arrives during A's tool wait: the slot is held for A.
        b = asyncio.create_task(scheduler.submit(request("B", 1)))
        await asyncio.sleep(0.01)
        self.assertEqual(backend.started_ids(), ["A-1"])
        second = await self.run_step(scheduler, backend, "A", 2)
        self.assertTrue(second["slot_hit"])
        self.assertEqual(second["slot_id"], 0)
        # B was skipped by the first hold, by A-2 jumping ahead, and by the
        # second hold: at the limit of 3 it takes the slot.
        await eventually(lambda: "B-1" in backend.started_ids())
        backend.release("B-1")
        result = await b
        self.assertFalse(result["slot_hit"])
        self.assertEqual(result["skips"], 3)

    async def test_hold_expires_when_agent_does_not_return(self):
        backend = SlotBackend()
        scheduler = await self.make(backend, slots=1, hold_s=0.05, max_skips=10)
        await self.run_step(scheduler, backend, "A", 1)
        b = asyncio.create_task(scheduler.submit(request("B", 1)))
        await eventually(lambda: "B-1" in backend.started_ids())
        backend.release("B-1")
        result = await b
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["skips"], 1)

    async def test_skip_limit_overrides_hold(self):
        backend = SlotBackend()
        scheduler = await self.make(backend, slots=1, hold_s=10, max_skips=2)
        await self.run_step(scheduler, backend, "A", 1)
        b = asyncio.create_task(scheduler.submit(request("B", 1)))
        await asyncio.sleep(0.01)
        # A's step 2 reuses the held slot and skips B a second time...
        await self.run_step(scheduler, backend, "A", 2)
        # ...so B reaches the limit and takes the slot despite the new hold.
        await eventually(lambda: "B-1" in backend.started_ids())
        backend.release("B-1")
        result = await b
        self.assertEqual(result["skips"], 2)
        self.assertEqual(backend.started_ids(), ["A-1", "A-2", "B-1"])

    async def test_agents_keep_their_slots_and_eviction_is_lru(self):
        backend = SlotBackend()
        scheduler = await self.make(backend, slots=2, hold_s=0, max_skips=3)
        await self.run_step(scheduler, backend, "A", 1)
        await self.run_step(scheduler, backend, "B", 1)
        slots = dict(backend.started)
        self.assertNotEqual(slots["A-1"], slots["B-1"])  # empty slot before eviction
        second = await self.run_step(scheduler, backend, "A", 2)
        self.assertTrue(second["slot_hit"])
        self.assertEqual(second["slot_id"], slots["A-1"])
        # C evicts the least recently used context, B's.
        third = await self.run_step(scheduler, backend, "C", 1)
        self.assertEqual(third["slot_id"], slots["B-1"])

    async def test_concurrency_never_exceeds_slots(self):
        backend = SlotBackend()
        scheduler = await self.make(backend, slots=3, hold_s=0)
        tasks = [asyncio.create_task(scheduler.submit(request(f"agent{i}", 1))) for i in range(8)]
        await eventually(lambda: len(backend.started) == 3)
        for i in range(8):
            backend.release(f"agent{i}-1")
            await tasks[i]
        results = await asyncio.gather(*tasks)
        self.assertEqual(backend.peak, 3)
        self.assertEqual({r["slot_id"] for r in results}, {0, 1, 2})
        self.assertTrue(all(r["status"] == "success" for r in results))

    async def test_timeout_and_failure_free_the_slot(self):
        class Failing:
            async def generate(self, value):
                if value["request_id"] == "A-1":
                    await asyncio.sleep(10)
                if value["request_id"] == "B-1":
                    raise RuntimeError("boom")
                return {"response": "ok"}

        scheduler = await self.make(Failing(), slots=1, hold_s=10, request_timeout_s=0.05)
        results = await asyncio.gather(
            scheduler.submit(request("A", 1)), scheduler.submit(request("B", 1)),
            scheduler.submit(request("C", 1)),
        )
        self.assertEqual([r["status"] for r in results], ["timeout", "failed", "success"])

    async def test_graceful_shutdown_drains_past_holds(self):
        backend = SlotBackend()
        scheduler = SlotAffinityScheduler(backend, slots=1, hold_s=10)
        await self.run_step(scheduler, backend, "A", 1)
        b = asyncio.create_task(scheduler.submit(request("B", 1)))
        await asyncio.sleep(0.01)
        shutdown = asyncio.create_task(scheduler.shutdown())
        await eventually(lambda: "B-1" in backend.started_ids())
        backend.release("B-1")
        await shutdown
        self.assertEqual((await b)["status"], "success")
        with self.assertRaises(RuntimeError):
            await scheduler.submit(request("C", 1))

    async def test_abort_cancels_running_and_waiting(self):
        backend = SlotBackend()
        scheduler = SlotAffinityScheduler(backend, slots=1, hold_s=0)
        tasks = [asyncio.create_task(scheduler.submit(request(f"agent{i}", 1))) for i in range(3)]
        await eventually(lambda: len(backend.started) == 1)
        await scheduler.shutdown(graceful=False)
        statuses = [r["status"] for r in await asyncio.gather(*tasks)]
        self.assertEqual(statuses, ["cancelled"] * 3)

    async def test_request_snapshot_is_not_mutated(self):
        backend = SlotBackend()
        scheduler = await self.make(backend, slots=1)
        value = request("A", 1)
        task = asyncio.create_task(scheduler.submit(value))
        await eventually(lambda: backend.started)
        backend.release("A-1")
        await task
        self.assertNotIn("id_slot", value["generation_settings"])

    def test_validation(self):
        for kwargs in ({"slots": 0}, {"slots": 1.5}, {"hold_s": -1}, {"hold_s": float("inf")},
                       {"max_skips": 0}, {"request_timeout_s": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SlotAffinityScheduler(object(), **kwargs)


if __name__ == "__main__":
    unittest.main()
