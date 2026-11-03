# KV cache metrics in `/metrics`

`llama-kv-metrics.patch` adds cache metrics to llama-server's `/metrics`
endpoint. `install_everything.sh` applies the patch before building.

```bash
curl -s localhost:8080/metrics | grep -E "kv_cache|prompt_cache|prompt_tokens"
```

All names below are prefixed with `llamacpp:` in the output. For background on
slots, the two cache levels and llama.cpp's cache policy, see
[kv_cache_notes.md](kv_cache_notes.md).

**Gauge vs counter.** A gauge is a snapshot of the current value and goes up
and down; sample it regularly (e.g. once per second) and report peak and
average. A counter only goes up from server start; take the difference (Δ)
between two readings to get what happened in between.

## GPU KV cache

| Metric | Type | What it represents |
| --- | --- | --- |
| `kv_cache_tokens` | gauge | Tokens stored in the GPU KV cache right now, summed over all slots. Includes idle slots, whose tokens are kept for prefix reuse. |
| `kv_cache_tokens_max` | gauge | Capacity of the GPU KV cache in tokens: the server's context size (`CTX_SIZE`). Fixed while the server runs. |
| `kv_cache_usage_ratio` | gauge | `kv_cache_tokens / kv_cache_tokens_max`. 0 = empty, 1 = full. |

## CPU RAM prompt cache

All are 0 if the prompt cache is disabled (`--cache-ram 0`). They also stay 0
until a slot's contents need to be saved, e.g. when an unrelated request takes
over a slot that already holds tokens.

| Metric | Type | What it represents |
| --- | --- | --- |
| `prompt_cache_entries` | gauge | Number of saved slot states in the prompt cache right now. One entry = one copy of a slot's KV data for one token sequence. |
| `prompt_cache_tokens` | gauge | Total tokens across all entries right now. Comparable to `kv_cache_tokens`, but for CPU RAM. |
| `prompt_cache_bytes` | gauge | CPU RAM used by the prompt cache right now, in bytes. This memory is part of the server's RSS. |
| `prompt_cache_bytes_max` | gauge | Size limit of the prompt cache in bytes (`--cache-ram`, default 8192 MiB). 0 means no limit. |
| `prompt_cache_saves_total` | counter | Times a slot's KV data was copied from the GPU into the prompt cache. Not counted if the same tokens were already saved. |
| `prompt_cache_loads_total` | counter | Times a saved entry was copied back from the prompt cache into a slot's GPU KV cache. The entry is removed from the prompt cache when this happens. |
| `prompt_cache_loaded_tokens_total` | counter | Over all loads, how many tokens at the start of the new prompt matched the restored entry. This is the part of `prompt_tokens_cached_total` that came from CPU RAM rather than tokens already on the GPU. |
| `prompt_cache_evictions_total` | counter | Entries deleted to stay under the size limit. The oldest entry is always deleted first. Entries removed because a newer, longer entry contains them are not counted. |

## Prompt token counters (built into llama.cpp)

| Metric | Type | What it represents |
| --- | --- | --- |
| `prompt_tokens_total` | counter | Prompt tokens the server actually computed (cache misses). |
| `prompt_tokens_cached_total` | counter | Prompt tokens reused instead of computed (cache hits), from either the GPU KV cache or a restored prompt cache entry. |
| `prompt_tokens_seconds` | gauge | Average prompt processing throughput in tokens/s. |

## GPU hits vs CPU RAM hits

A GPU hit reuses tokens already in a slot; a CPU RAM hit first copies a saved
entry back to the GPU, which costs extra time. Over a time window:

| Question | How to get it |
| --- | --- |
| Tokens that hit on the GPU | Δ `prompt_tokens_cached_total` − Δ `prompt_cache_loaded_tokens_total` |
| Tokens that hit in CPU RAM | Δ `prompt_cache_loaded_tokens_total` |
| Number of RAM-hit events (restores) | Δ `prompt_cache_loads_total` |
| Number of GPU-hit events | Not in `/metrics`; see per-request counts below |
| Overall hit rate | Δ `prompt_tokens_cached_total` / (Δ `prompt_tokens_cached_total` + Δ `prompt_tokens_total`) |

These are approximate: when a prompt fully matches the cache, the server still
recomputes its last token, so `prompt_cache_loaded_tokens_total` can be
slightly higher than the hits it actually produced.

**Per-request counts.** Every completion response reports its own cached
tokens: `usage.prompt_tokens_details.cached_tokens` (OpenAI-style) or
`timings.cache_n` (llama.cpp's field). Log it next to each request's latency
to see which requests hit.

## Recording over time with `monitor.py`

`monitor.py` samples once per `MONITOR_INTERVAL` seconds (default 1) and writes
one CSV row per sample to `MONITOR_OUTPUT` (default `gpu_system_metrics.csv`).
It reads `/metrics` from `SERVER_URL` (default `http://127.0.0.1:8080`).

```bash
python inference_setup/monitor.py  # Ctrl-C to stop
```

| Column | Source | What it represents |
| --- | --- | --- |
| `utc_time` | clock | Sample time (UTC) |
| `gpu_memory_used_mib`, `gpu_memory_total_mib` | nvidia-smi | GPU memory used by all processes / GPU total |
| `gpu_utilization_pct` | nvidia-smi | GPU compute utilization |
| `server_gpu_memory_mib` | nvidia-smi | GPU memory used by the llama-server process alone (weights + KV cache + buffers); empty if nvidia-smi can't attribute it |
| `system_available_bytes`, `system_used_bytes` | psutil | Machine-wide CPU RAM available / used |
| `server_rss_bytes` | psutil | CPU RAM used by the llama-server process; includes `prompt_cache_bytes` |
| `prompt_tokens_total`, `prompt_tokens_cached_total` | `/metrics` | Prefix cache misses / hits (counters) |
| `kv_cache_tokens`, `kv_cache_tokens_max`, `kv_cache_usage_ratio` | `/metrics` | GPU KV cache occupancy |
| `prompt_cache_entries`, `prompt_cache_tokens`, `prompt_cache_bytes`, `prompt_cache_bytes_max` | `/metrics` | CPU RAM prompt cache size |
| `prompt_cache_saves_total`, `prompt_cache_loads_total`, `prompt_cache_loaded_tokens_total`, `prompt_cache_evictions_total` | `/metrics` | CPU RAM prompt cache activity (counters) |
| `requests_processing`, `requests_deferred` | `/metrics` | Requests running in a slot / waiting for a free slot |

`/metrics` columns are empty for a sample if the server didn't answer. For the
counters, compute the per-interval change between consecutive rows (e.g.
`df[col].diff()` in pandas) and plug it into the hit-rate formulas above.

## Troubleshooting

- If `grep kv_cache` or `grep prompt_cache` prints nothing (or `monitor.py`
  warns "no kv_cache metrics"), the running server is an older build. Rerun
  `install_everything.sh` and restart `start_server.sh`.
- The patch also makes `/metrics` print values with 15 significant digits.
  llama.cpp's default of 6 would round large counters, e.g. 1234567 to
  `1.23457e+06`.
- The patch is tested at the llama.cpp commit in `llama_cpp_commit.txt`. It
  changes `tools/server/server-task.h`, `server-task.cpp` and
  `server-context.cpp`; `install_everything.sh` resets those three files before
  applying it, so manual edits to them are discarded.
