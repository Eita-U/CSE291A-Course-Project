"""Slot-affinity dispatch: keep each agent's context in the same llama.cpp slot.

llama.cpp reuses a prompt prefix only while it is still in a slot. With more
agents than slots, FIFO hands a freed slot to whichever request is next, which
overwrites the previous agent's context. This scheduler instead:

1. Pins every request to a server slot with ``id_slot`` and remembers which
   agent (task_id) each slot holds.
2. Sends a waiting request to its agent's slot when that slot is free.
3. After a request finishes, holds its slot for ``hold_s`` seconds so the
   agent's next step, which arrives after a tool wait, finds its context.
4. Gives a slot to an agent without one only by evicting an empty slot first,
   then the least recently used one, never a held slot.
5. Bounds unfairness: a request skipped ``max_skips`` times (a younger
   request dispatched first, or a slot held while it waited) takes the next
   free slot, held or not.

``slots`` must equal the server's ``--parallel``; slot i maps to id_slot i.
"""

import asyncio
import math
from dataclasses import dataclass, field
from time import perf_counter as _perf
from typing import Any

from .fifo import LLMBackend, _build_response, _Pending, _utc_now, _validate_result
from .schemas import InferenceRequest, InferenceResponse, RequestStatus


@dataclass
class _Slot:
    index: int
    busy: bool = False
    owner: str | None = None  # task whose context the slot holds
    last_used: float = 0.0
    hold_until: float = 0.0   # reserved for the owner's next step until then
    charged: set[str] = field(default_factory=set)  # requests already skipped by this hold


@dataclass
class _Waiting:
    item: _Pending
    skips: int = 0


class SlotAffinityScheduler:
    """Single-event-loop, single-use scheduler with the FIFOScheduler interface.

    Results add ``slot_id``, ``slot_hit`` (sent to the slot already holding its
    agent's context) and ``skips`` to the FIFO response fields.
    """

    def __init__(
        self, backend: LLMBackend, slots: int = 1,
        request_timeout_s: float | None = 60.0,
        hold_s: float = 2.0, max_skips: int = 3,
    ) -> None:
        if type(slots) is not int or slots < 1:
            raise ValueError("slots must be a positive integer")
        if request_timeout_s is not None and (
            isinstance(request_timeout_s, bool)
            or not isinstance(request_timeout_s, (int, float))
            or not math.isfinite(request_timeout_s)
            or request_timeout_s <= 0
        ):
            raise ValueError("request_timeout_s must be positive and finite, or None")
        if (
            isinstance(hold_s, bool) or not isinstance(hold_s, (int, float))
            or not math.isfinite(hold_s) or hold_s < 0
        ):
            raise ValueError("hold_s must be a non-negative finite number")
        if type(max_skips) is not int or max_skips < 1:
            raise ValueError("max_skips must be a positive integer")
        self.backend = backend
        self.request_timeout_s = request_timeout_s
        self.hold_s = float(hold_s)
        self.max_skips = max_skips
        self._slots = [_Slot(i) for i in range(slots)]
        self._waiting: list[_Waiting] = []
        self._pending: dict[str, _Pending] = {}
        self._running: dict[int, asyncio.Task[None]] = {}
        self._timers: set[asyncio.TimerHandle] = set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._closing = False
        self._aborting = False
        self._shutdown_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def concurrency(self) -> int:
        return len(self._slots)

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("Scheduler must be used on its original event loop")
        self._loop = loop

    async def start(self) -> None:
        self._check_loop()
        if self._closing:
            raise RuntimeError("Scheduler is shutting down or closed")

    async def submit(
        self, request: InferenceRequest | dict[str, Any],
    ) -> InferenceResponse:
        parsed = InferenceRequest.parse(request)
        await self.start()
        if parsed.request_id in self._pending:
            raise ValueError(f"Duplicate outstanding request_id: {parsed.request_id}")
        item = _Pending(
            parsed, asyncio.get_running_loop().create_future(),
            _perf(), _utc_now(),
        )
        self._pending[parsed.request_id] = item
        self._waiting.append(_Waiting(item))
        self._idle.clear()
        self._dispatch()
        return await asyncio.shield(item.future)

    def get_status(self, request_id: str) -> str | None:
        """Return queued/running for outstanding work; completed records are returned to callers."""
        item = self._pending.get(request_id)
        return None if item is None else ("queued" if item.dispatched_at is None else "running")

    # Dispatch policy. Times are event-loop time so hold timers and checks agree.

    def _home(self, entry: _Waiting) -> _Slot | None:
        """The free slot holding this request's agent context, if any."""
        task_id = entry.item.request.task_id
        return next(
            (s for s in self._slots if s.owner == task_id and not s.busy), None,
        )

    def _victim(self, free: list[_Slot], now: float, allow_held: bool) -> _Slot | None:
        """Free slot to overwrite: empty first, then least recently used."""
        candidates = [s for s in free if allow_held or s.hold_until <= now]
        if not candidates:
            return None
        return min(candidates, key=lambda s: (s.owner is not None, s.last_used))

    def _choose(self, free: list[_Slot], now: float) -> tuple[int, _Slot] | None:
        for i, entry in enumerate(self._waiting):
            if entry.skips >= self.max_skips:
                slot = self._home(entry) or self._victim(free, now, allow_held=True)
                return i, slot
        for i, entry in enumerate(self._waiting):
            home = self._home(entry)
            if home is not None:
                return i, home
        slot = self._victim(free, now, allow_held=False)
        return None if slot is None else (0, slot)

    def _dispatch(self) -> None:
        if self._aborting:
            return
        now = asyncio.get_running_loop().time()
        while self._waiting:
            free = [s for s in self._slots if not s.busy]
            if not free:
                break
            choice = self._choose(free, now)
            if choice is None:
                # Every free slot is held. Each hold skips the request next in
                # line once; at the limit it takes a held slot.
                head = self._waiting[0]
                rid = head.item.request.request_id
                for slot in free:
                    if rid not in slot.charged:
                        slot.charged.add(rid)
                        head.skips += 1
                if head.skips < self.max_skips:
                    break
                continue
            index, slot = choice
            entry = self._waiting.pop(index)
            for older in self._waiting[:index]:
                older.skips += 1
            self._launch(entry, slot)

    def _launch(self, entry: _Waiting, slot: _Slot) -> None:
        item = entry.item
        task_id = item.request.task_id
        hit = slot.owner == task_id
        for other in self._slots:
            if other is not slot and other.owner == task_id:
                other.owner = None  # stale copy; the agent's context moves to slot
        slot.busy, slot.owner, slot.hold_until = True, task_id, 0.0
        item.worker_id = slot.index + 1
        item.dispatched_at, item.dispatched_utc = _perf(), _utc_now()
        self._running[slot.index] = asyncio.create_task(
            self._run(entry, slot, hit), name=f"slot-{slot.index}",
        )

    def _release(self, slot: _Slot, hold: bool) -> None:
        loop = asyncio.get_running_loop()
        now = loop.time()
        slot.busy, slot.last_used = False, now
        self._running.pop(slot.index, None)
        if hold and self.hold_s > 0 and not self._closing:
            slot.hold_until = now + self.hold_s
            slot.charged.clear()
            self._timers = {t for t in self._timers if t.when() > now}
            self._timers.add(loop.call_later(self.hold_s, self._dispatch))
        self._dispatch()
        if not self._waiting and not self._running:
            self._idle.set()

    # Execution

    async def _run(self, entry: _Waiting, slot: _Slot, hit: bool) -> None:
        item = entry.item
        request = item.request.to_dict()
        request["generation_settings"]["id_slot"] = slot.index
        extra = {"slot_id": slot.index, "slot_hit": hit, "skips": entry.skips}
        status: RequestStatus = "failed"
        try:
            async with asyncio.timeout(self.request_timeout_s):
                result = await self.backend.generate(request)
            usage = _validate_result(result)
        except TimeoutError as exc:
            self._finish(item, "timeout", error=str(exc) or "Backend request timed out", extra=extra)
        except asyncio.CancelledError:
            if self._aborting:
                self._finish(item, "cancelled", error="Scheduler aborted", extra=extra)
                raise
            self._finish(item, "failed", error="Backend cancelled the request", extra=extra)
        except Exception as exc:
            self._finish(item, "failed", error=f"{type(exc).__name__}: {exc}", extra=extra)
        else:
            status = "success"
            self._finish(
                item, "success", result["response"], usage,
                result.get("backend_timings", {}), extra=extra,
            )
        finally:
            if not self._aborting:
                # An agent whose step failed stops, so only hold after success.
                self._release(slot, hold=status == "success")

    def _finish(
        self, item: _Pending, status: RequestStatus, response: str | None = None,
        token_usage: dict[str, int] | None = None, backend_timings: dict[str, Any] | None = None,
        error: str | None = None, extra: dict[str, Any] | None = None,
    ) -> None:
        result = _build_response(item, status, response, token_usage, backend_timings, error)
        result.update(extra or {"slot_id": None, "slot_hit": False, "skips": 0})
        self._pending.pop(item.request.request_id, None)
        if not item.future.done():
            item.future.set_result(result)

    async def shutdown(self, *, graceful: bool = True) -> None:
        """Reject new work and drain, or abort with explicit cancelled results.

        Idempotent. The first shutdown call selects the mode. Cleanup continues
        if a caller waiting for shutdown is itself cancelled.
        """
        self._check_loop()
        if self._shutdown_task is None:
            self._closing = True
            self._aborting = not graceful
            self._shutdown_task = asyncio.create_task(self._shutdown(graceful))
        await asyncio.shield(self._shutdown_task)

    async def _shutdown(self, graceful: bool) -> None:
        if graceful:
            # Holds would only delay draining; let waiting work take any slot.
            for slot in self._slots:
                slot.hold_until = 0.0
            self._dispatch()
            await self._idle.wait()
        else:
            running = list(self._running.values())
            for task in running:
                task.cancel()
            for entry in self._waiting:
                self._finish(entry.item, "cancelled", error="Scheduler aborted before dispatch")
            self._waiting.clear()
            await asyncio.gather(*running, return_exceptions=True)
            self._running.clear()
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()

    async def __aenter__(self) -> "SlotAffinityScheduler":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.shutdown(graceful=exc_type is None)
