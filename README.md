# CSE291A Course Project

Phase 1 FIFO baseline for memory-aware agent serving. This repository contains
the FIFO scheduler, the synthetic microbenchmark harness, the Qwen HTTP
backend, and the post-run memory analysis scripts. llama.cpp still runs on the
GPU instance; this repo does not start that process.

The recorded baseline is `results/phase1-fifo-005`: three trials, Qwen3-1.7B
Q8_0, `--ctx-size 4096 --parallel 2`. Admitted suites finished with no
timeouts or connection errors. `oversize_context` is the only expected failure
(HTTP 400 when a 2048-word prompt exceeds the 2048-token slot). Earlier
directories (`phase1-fifo-002` through `004`) are not the baseline.

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
instance.

Local `peak_benchmark_process_rss_bytes` is the **benchmark Python process
RSS**. GPU VRAM, llama-server RSS, KV occupancy, and prompt-cache counters
come from an independent EC2 CSV, `gpu_system_metrics.csv`, joined to
`suite_boundary_events.jsonl` by UTC. Use the same directory for `--out` and
the same string for `--run-id`. This harness does not check clock sync.

`--run-id` must match `--out`. A mismatch leaves `run_id` inside the JSONL
different from the folder name; `scripts/analyze_memory.py` still reads the
folder, and prints a warning.

To run against the real Qwen server (already started on the GPU instance, or
SSH-tunneled to `127.0.0.1:8080`), pass `--backend qwen`. That injects
`QwenHTTPBackend` through the existing `backend_factory`; it does not start
the model process.

Default **long** prompts are 512 whitespace tokens so they fit one llama.cpp
slot at `--ctx-size 4096 --parallel 2`. 2048-word pads belong only to the
`oversize_context` admission suite (expected HTTP 400), not the mixed or
concurrency matrices. Metrics include an all-request view and
`successful_only` latency/throughput. Override with `--long-context-tokens`
and `--oversize-context-tokens`.

Python **3.11+**. Tests and the scheduler use the standard library. Graphs need
matplotlib (`pip install -r requirements-bench.txt`). Real-model runs also need
httpx (`pip install -r requirements-backend.txt`). From the repository root:

```bash
python3 -m unittest discover -v
python3 -m examples.basic_usage
python3 -m examples.workload_run
python3 -m examples.run_microbenchmarks --quick
python3 -m examples.run_microbenchmarks --out results/microbenchmarks
python3 -m examples.run_microbenchmarks --backend qwen --quick
python3 -m examples.run_microbenchmarks --backend qwen \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --out results/phase1-fifo-005 --run-id phase1-fifo-005 \
  --trials 3 --warmup
```

Start the EC2 monitor before the first suite and leave it running until the
last suite finishes. Copy `gpu_system_metrics.csv` into the same `--out`
directory, then:

```bash
python3 scripts/analyze_memory.py --run-id phase1-fifo-005
python3 scripts/summarize_microbenchmark.py --run-id phase1-fifo-005
python3 scripts/bundle_memory_analysis.py --run-id phase1-fifo-005
```

`analyze_memory.py` requires `utc_time`, `gpu_memory_used_mib`,
`gpu_memory_total_mib`, `gpu_utilization_pct`, and `server_rss_bytes`. KV
tokens, prompt-cache fields, and `server_gpu_memory_mib` are optional; empty
cells after the server exits are skipped. The checked-in
`inference_setup/monitor.py` records nvidia-smi and host RAM only. The Phase 1
CSV was produced by the expanded EC2 monitor that also samples llama.cpp
metrics. `summarize_microbenchmark.py` writes `analysis_brief.md` (suite
medians from healthy trials; oversize rejects stay in the failure counts).
`bundle_memory_analysis.py` copies that brief plus the memory tables into
`memory_analysis_bundle/` and leaves traces and the 1 Hz CSV out.

## Realistic Macrobenchmarks (Task 4)

Prepare `data/processed/trace_32k.parquet` following [data/README.md](data/README.md).
Install replay and analysis dependencies in your Python environment:

```bash
python -m pip install pandas pyarrow numpy tqdm httpx
python -m pip install psutil requests  # EC2 memory monitor
```

In `examples/run_trace.py`, set the session selection, `concurrency`,
`pre_gap_scale`, `cache_prompt`, and output `run_id`. The current example selects
three sessions with seed 42 and replays one at a time using `task_index` (0–2).
Steps use recorded conversation histories; tool delays are simulated with
`pre_gap`, and recorded output lengths become generation limits.

Run from the repository root on EC2, using separate terminals for the server,
monitor, and replay. Follow [inference_setup/README.md](inference_setup/README.md)
for server installation. Example with concurrency 1 and 32K context per slot:

```bash
# Terminal 1: start a fresh inference server for each condition.
CTX_SIZE=32768 PARALLEL=1 bash inference_setup/start_server.sh

# Terminal 2: after the server is ready, start monitoring.
MONITOR_INTERVAL=0.2 MONITOR_OUTPUT=measurement/memory.csv python inference_setup/monitor.py

# Terminal 3: replay the selected sessions.
python -m examples.run_trace
```

Match server `PARALLEL` to FIFO `concurrency` and set `CTX_SIZE` to
`32768 * PARALLEL`. Confirm the monitor PID matches the intended server.
After replay finishes, stop the monitor with Ctrl-C. Set the result JSON and
matching CSV paths in `examples/run_trace_calculation.py`, then run:

```bash
python -m examples.run_trace_calculation
```

Files produced by these scripts:

| File | Purpose and location |
| --- | --- |
| `results.json` | Configuration, run/task timing, request outcomes, token counts, and backend timings; written to `measurement/results/<run_id>/results.json`. |
| `memory.csv` | Timestamped GPU memory, server RSS, and cache metrics; written to `measurement/memory.csv` by the command above. |
| `metrics.json` | Aggregated performance, memory, prefix reuse, and per-step context-growth metrics; written to `measurement/metrics.json`. |

Archive the matching CSV and metrics beside each run's `results.json` before
the next run overwrites the shared output files. Analysis selects memory
samples within the recorded UTC execution windows. Collected experiments are
organized under:

- `measurement/results/concurrency/`: 20 sessions, concurrency 1/2/4/5.
- `measurement/results/idle/`: five sessions, delay scales 0/1/2.
- `measurement/results/prefix/`: three sessions replayed individually with
  prefix reuse ON/OFF, also used for context-growth analysis.

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
active. The Phase 1 server uses `--parallel 2`, so FIFO concurrency 2 matches
the slot count and concurrency 4 waits inside llama.cpp. Live calls go through
`scheduler/qwen_backend.py` (`QwenHTTPBackend`); the server must already be
listening, usually via the SSH tunnel in `inference_setup/README.md`.

## Layout and verification

- `scheduler/fifo.py`: admission, worker pool, timing, and shutdown.
- `scheduler/schemas.py`: validated request and typed response contract.
- `scheduler/mock_llm.py`: local async mock; no external services.
- `scheduler/qwen_backend.py`: HTTP client for an already-running llama.cpp server.
- `workload/`: synthetic profiles, dummy-agent runner, suite metrics, and plots.
- `examples/run_microbenchmarks.py`: mock or Qwen suite runner.
- `inference_setup/`: EC2 install, server start, smoke test, and the basic monitor.
- `scripts/analyze_memory.py`: join the EC2 CSV to suite windows.
- `scripts/summarize_microbenchmark.py`: assignment-sized tables from trial metrics.
- `scripts/bundle_memory_analysis.py`: high-level memory bundle, without traces.
- `tests/`: FIFO ordering, workload construction, Qwen HTTP mapping, and suite checks.

The queue and backend execution are separated so a later policy can replace
admission and dispatch. Phase 2 memory-aware policies are outside this baseline.
