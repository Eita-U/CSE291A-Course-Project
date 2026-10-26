# CSE291A Course Project

Phase 1: FIFO baseline for memory-aware agent serving. This contribution
implements Yuchen's request scheduler only. AWS deployment, HTTP transport,
agent workload generation, memory monitoring, and experiment analysis remain
separate team components.

## Run locally

Python **3.11+**; runtime and tests use only the standard library. From the
repository root:

```bash
python3 -m unittest discover -v
python3 -m examples.basic_usage
```

## FIFO microbenchmarks (Task 3)

Workloads, dummy agents, request/task traces, and plots sit **on top of**
`FIFOScheduler`. The scheduler is not modified. Default runs use
`WorkloadMockLLM` on the local event loop: no GPU, no Qwen process, no AWS
instance. Peak memory is **process RSS**, not inference/GPU memory. Swap in a
real `LLMBackend` factory when the HTTP inference server exists.

Python **3.11+**. Tests and the scheduler use the standard library. Graphs need
matplotlib (`pip install -r requirements-bench.txt`). From the repository root:

```bash
python3 -m unittest discover -v
python3 -m examples.basic_usage
python3 -m examples.workload_run
python3 -m examples.run_microbenchmarks --quick
python3 -m examples.run_microbenchmarks --out results/microbenchmarks

## Scheduler interface

```python
from scheduler import FIFOScheduler
from scheduler.mock_llm import MockLLMBackend

async def infer():
    async with FIFOScheduler(
        MockLLMBackend(), concurrency=2, request_timeout_s=60,
    ) as scheduler:
        result = await scheduler.submit({
            "request_id": "req_001",
            "task_id": "agent_001",
            "step_id": 2,
            "prompt": "Analyze the following document...",
            "generation_settings": {"max_tokens": 128, "temperature": 0},
        })
        return result
```

Share **one scheduler** across concurrent agent tasks. The
[minimal runner example](examples/basic_usage.py) demonstrates two independent
two-step tasks; simulated tool waits occur outside `submit` and occupy no worker.
Use an async context manager, or call `await scheduler.shutdown()` in `finally`.
`submit` starts the workers automatically; `start()` is also available and
idempotent before shutdown.

### Request contract

`submit` accepts a dictionary or `InferenceRequest` with these fields:

The team's Task 2 contract is authoritative: `step_id` is required even though
the abbreviated Task 3 note omits it.

| Field | Type | Meaning |
| --- | --- | --- |
| `request_id` | nonempty string | Unique among outstanding requests |
| `task_id` | nonempty string | Agent task identifier |
| `step_id` | nonnegative integer | Step in the agent workflow |
| `prompt` | string | Prompt built by the agent runner |
| `generation_settings` | dictionary, default `{}` | Opaque parameters passed to the backend |

Malformed requests and duplicate outstanding IDs raise `ValueError` before
enqueue. Unknown top-level fields are rejected; `messages` is not yet part of
this contract. The request is copied at acceptance so later caller mutations
cannot change queued work. Model-specific generation validation belongs to the
backend. IDs may be reused after completion, but experiment runners should use
globally unique IDs for unambiguous traces.

### Backend contract (Chengjun)

Inject any object exposing this asynchronous method:

```python
async def generate(self, request: dict) -> dict:
    # Translate prompt/settings to the inference server's wire format here.
    return {
        "response": "Generated text",
        "token_usage": {"prompt_tokens": 512, "completion_tokens": 120},
    }
```

`response` is required and must be a string. `token_usage` is optional; when
provided it must map strings to nonnegative integers. Missing usage becomes
`{}` rather than invented token counts. Backend exceptions and malformed backend
results become failed responses without killing a worker. A backend
`TimeoutError` becomes a timeout response. The scheduler never retries requests.
The HTTP client owns its connection pool and must be closed by its owner after
the scheduler has shut down.

Backend code must use nonblocking async I/O and cooperate with cancellation.
Timeouts use `asyncio.timeout`; cancellation-suppressing backends cannot provide
reliable timeout or shutdown guarantees. Set `request_timeout_s=None` to disable
the scheduler timeout. The default is 60 seconds **from dispatch**, excluding
scheduler queue wait. HTTP-specific timeout controls belong to the client.

### Response and timing contract (Dillon / Eita)

Every accepted request returns a JSON-compatible dictionary:

```json
{
  "request_id": "req_001",
  "task_id": "agent_001",
  "step_id": 2,
  "response": "Generated text",
  "status": "success",
  "worker_id": 1,
  "timing_metadata": {
    "queue_entry_time": "2026-10-21T20:00:00Z",
    "execution_start_time": "2026-10-21T20:00:00.150Z",
    "execution_finish_time": "2026-10-21T20:00:02Z",
    "queue_time_ms": 150.0,
    "api_time_ms": 1850.0,
    "total_time_ms": 2000.0
  },
  "token_usage": {"prompt_tokens": 512, "completion_tokens": 120},
  "error": null
}
```

Terminal statuses are `success`, `failed`, `timeout`, and `cancelled`. Unsuccessful
responses have `response: null`, empty usage, and an error string. Workers use
one-based IDs. `get_status(request_id)` exposes `queued` or `running` for
outstanding work, and returns `None` after completion or for an unknown ID.
The scheduler retains no completed-result history; the runner should persist
returned records for evaluation.

The plan's required timing fields are `queue_entry_time`,
`execution_start_time`, and `execution_finish_time`, all UTC timestamps.
`execution_start_time` marks scheduler dispatch into the backend;
`execution_finish_time` marks completion, failure, timeout, or cancellation.
They do not identify model-only GPU execution boundaries. The old names
`dispatch_time` and `finish_time` are not emitted.

`worker_id`, `error`, and the three duration fields are additional diagnostic
fields beyond the plan's required contract. Durations use `time.perf_counter()`:

| Metric | Definition |
| --- | --- |
| `queue_time_ms` | Dispatch minus queue entry |
| `api_time_ms` | Finish minus dispatch, including backend overhead/server queueing |
| `total_time_ms` | Finish minus queue entry |

Dispatch marks entry into backend processing, not the start of model computation.
For an aborted queued request, `worker_id` and `execution_start_time` are `null`, API
time is zero, and queue time equals total time until cancellation. Aborted
running requests retain dispatch and worker metadata.

The runner measures task-level completion and tool waits. Server-side monitoring
must measure inference memory. Scheduler durations do not measure model-only
execution or peak inference memory.

## Scheduling and shutdown semantics

- An `asyncio.Queue` determines FIFO **dispatch**, not completion order. Arrival
  is the order in which `submit` coroutines execute their enqueue operation.
- Exactly `concurrency` workers are created; each awaits at most one backend
  call at a time. Configure a new scheduler for each benchmark concurrency.
- The queue is unbounded for this baseline. The scheduler and backend run on
  one event loop, and a scheduler cannot restart after shutdown.
- `shutdown()` rejects new submissions, finishes all accepted work, and joins
  workers. A normal context exit does the same. With timeout disabled, draining
  waits indefinitely if a backend never completes.
- `shutdown(graceful=False)` cancels running backend calls and resolves all
  accepted requests with explicit cancelled results. Exceptional context exit
  uses this mode. Already completed requests keep their outcomes.
- Shutdown is idempotent; the first call selects its mode. Cancelling a shutdown
  waiter leaves cleanup running; await `shutdown()` again to join it.
- Cancelling a `submit` waiter leaves accepted work running. Its result is no
  longer delivered to that cancelled waiter. To collect every experiment record,
  keep submit tasks alive until terminal outcomes are returned.

The fixed HTTP-call limit does not establish how many llama.cpp model slots are
active. Coordinate server parallel-slot configuration separately. No HTTP API
adapter or live inference integration is included in this component.

## Layout and verification

- `scheduler/schemas.py`: validated request and typed response contract.
- `scheduler/fifo.py`: admission, worker pool, timing, and shutdown.
- `scheduler/mock_llm.py`: local async mock; no external services.
- `tests/test_fifo.py`: event-gated ordering/concurrency, response mapping,
  monotonic timing, failures, timeouts, cancellation, validation, and shutdown.
- `examples/basic_usage.py`: minimal caller integration.

The queue and backend execution are separated in the implementation so a future
policy can replace admission/dispatch behavior. Phase 2 memory-aware policies
are intentionally outside this baseline.
