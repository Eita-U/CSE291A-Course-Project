"""FIFO dispatch and fixed concurrency, independent of the inference transport."""

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from .schemas import InferenceRequest, InferenceResponse, RequestStatus


class LLMBackend(Protocol):
    async def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return {'response': str, 'token_usage': optional dict[str, int]}.

        Exceptions represent backend failures. Implementations must be async and
        cooperate with cancellation; do not swallow asyncio.CancelledError.
        """
        ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class _Pending:
    request: InferenceRequest
    future: asyncio.Future[InferenceResponse]
    queued_at: float
    queued_utc: str
    dispatched_at: float | None = None
    dispatched_utc: str | None = None
    worker_id: int | None = None


class FIFOScheduler:
    """Single-event-loop, single-use scheduler with an unbounded FIFO queue.

    Lifecycle: queued -> running -> success/failed/timeout, or cancelled during
    abort. Default shutdown drains accepted work. Cancelling a submit waiter
    does not cancel accepted inference; use shutdown(graceful=False) to abort.
    Request IDs must be unique among outstanding requests.
    """

    def __init__(
        self, backend: LLMBackend, concurrency: int = 1,
        request_timeout_s: float | None = 60.0,
    ) -> None:
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        if request_timeout_s is not None and (
            isinstance(request_timeout_s, bool)
            or not isinstance(request_timeout_s, (int, float))
            or not math.isfinite(request_timeout_s)
            or request_timeout_s <= 0
        ):
            raise ValueError("request_timeout_s must be positive and finite, or None")
        self.backend = backend
        self._concurrency = concurrency
        self.request_timeout_s = request_timeout_s
        self._queue: asyncio.Queue[_Pending | None] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._pending: dict[str, _Pending] = {}
        self._closing = False
        self._aborting = False
        self._shutdown_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def concurrency(self) -> int:
        return self._concurrency

    def _check_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("Scheduler must be used on its original event loop")
        self._loop = loop

    async def start(self) -> None:
        self._check_loop()
        if self._closing:
            raise RuntimeError("Scheduler is shutting down or closed")
        if not self._workers:
            self._workers = [
                asyncio.create_task(self._worker(i), name=f"fifo-worker-{i}")
                for i in range(1, self.concurrency + 1)
            ]

    async def submit(
        self, request: InferenceRequest | dict[str, Any],
    ) -> InferenceResponse:
        parsed = InferenceRequest.parse(request)
        await self.start()
        if parsed.request_id in self._pending:
            raise ValueError(f"Duplicate outstanding request_id: {parsed.request_id}")
        item = _Pending(
            parsed, asyncio.get_running_loop().create_future(),
            time.perf_counter(), _utc_now(),
        )
        # No await between the admission check and enqueue: shutdown cannot
        # interleave with acceptance, and put_nowait defines arrival order.
        self._pending[parsed.request_id] = item
        self._queue.put_nowait(item)
        return await asyncio.shield(item.future)

    def get_status(self, request_id: str) -> str | None:
        """Return queued/running for outstanding work; completed records are returned to callers."""
        item = self._pending.get(request_id)
        return None if item is None else ("queued" if item.dispatched_at is None else "running")

    def _finish(
        self, item: _Pending, status: RequestStatus, response: str | None = None,
        token_usage: dict[str, int] | None = None, error: str | None = None,
    ) -> None:
        finished_at, finished_utc = time.perf_counter(), _utc_now()
        dispatch = item.dispatched_at
        result: InferenceResponse = {
            "request_id": item.request.request_id,
            "task_id": item.request.task_id,
            "step_id": item.request.step_id,
            "response": response, "status": status,
            "worker_id": item.worker_id,
            "timing_metadata": {
                "queue_entry_time": item.queued_utc,
                "dispatch_time": item.dispatched_utc,
                "finish_time": finished_utc,
                "queue_time_ms": ((dispatch if dispatch is not None else finished_at) - item.queued_at) * 1000,
                "api_time_ms": 0.0 if dispatch is None else (finished_at - dispatch) * 1000,
                "total_time_ms": (finished_at - item.queued_at) * 1000,
            },
            "token_usage": dict(token_usage or {}), "error": error,
        }
        self._pending.pop(item.request.request_id, None)
        if not item.future.done():
            item.future.set_result(result)

    async def _worker(self, worker_id: int) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                if self._aborting:
                    self._finish(item, "cancelled", error="Scheduler aborted before dispatch")
                    continue
                item.worker_id = worker_id
                item.dispatched_at, item.dispatched_utc = time.perf_counter(), _utc_now()
                try:
                    # Direct awaiting preserves FIFO entry into generate(),
                    # including when the backend completes without suspending.
                    async with asyncio.timeout(self.request_timeout_s):
                        result = await self.backend.generate(item.request.to_dict())
                    if not isinstance(result, dict) or not isinstance(result.get("response"), str):
                        raise ValueError("Backend must return a dictionary with a string response")
                    usage = result.get("token_usage", {})
                    if not isinstance(usage, dict) or any(
                        not isinstance(k, str) or type(v) is not int or v < 0
                        for k, v in usage.items()
                    ):
                        raise ValueError("Backend token_usage must contain non-negative integer counts")
                except TimeoutError as exc:
                    self._finish(item, "timeout", error=str(exc) or "Backend request timed out")
                except asyncio.CancelledError:
                    if self._aborting:
                        self._finish(item, "cancelled", error="Scheduler aborted")
                        raise
                    self._finish(item, "failed", error="Backend cancelled the request")
                except Exception as exc:
                    self._finish(item, "failed", error=f"{type(exc).__name__}: {exc}")
                else:
                    self._finish(item, "success", result["response"], usage)
            finally:
                self._queue.task_done()

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
            await self._queue.join()
            for _ in self._workers:
                self._queue.put_nowait(None)
        else:
            for worker in self._workers:
                worker.cancel()
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if item is not None:
                    self._finish(item, "cancelled", error="Scheduler aborted before dispatch")
                self._queue.task_done()
        await asyncio.gather(*self._workers, return_exceptions=True)

    async def __aenter__(self) -> "FIFOScheduler":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.shutdown(graceful=exc_type is None)
