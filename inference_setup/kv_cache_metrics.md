# KV cache metrics in `/metrics`

`llama-kv-metrics.patch` adds cache metrics to llama-server's `/metrics`
endpoint. `install_everything.sh` applies the patch before building.

```bash
curl -s localhost:8080/metrics | grep -E "kv_cache|prompt_cache|prompt_tokens"
```

All names below are prefixed with `llamacpp:` in the output. A **gauge** is a
current value that goes up and down; a **counter** only goes up from server
start (see [Gauges vs counters](#gauges-vs-counters)).

### GPU KV cache

| Metric | Type | What it represents |
| --- | --- | --- |
| `kv_cache_tokens` | gauge | Tokens stored in the GPU KV cache right now, summed over all slots. Includes idle slots, whose tokens are kept for prefix reuse. |
| `kv_cache_tokens_max` | gauge | Capacity of the GPU KV cache in tokens: the server's context size (`CTX_SIZE`). Fixed while the server runs. |
| `kv_cache_usage_ratio` | gauge | `kv_cache_tokens / kv_cache_tokens_max`. 0 = empty, 1 = full. |

### CPU RAM prompt cache

All are 0 if the prompt cache is disabled (`--cache-ram 0`).

| Metric | Type | What it represents |
| --- | --- | --- |
| `prompt_cache_entries` | gauge | Number of saved slot states in the prompt cache right now. One entry = one copy of a slot's KV data for one token sequence. |
| `prompt_cache_tokens` | gauge | Total tokens across all entries right now. Comparable to `kv_cache_tokens`, but for CPU RAM. |
| `prompt_cache_bytes` | gauge | CPU RAM used by the prompt cache right now, in bytes. This memory is part of the server's RSS. |
| `prompt_cache_bytes_max` | gauge | Size limit of the prompt cache in bytes (`--cache-ram`, default 8192 MiB). 0 means no limit. |
| `prompt_cache_saves_total` | counter | Times a slot's KV data was copied from the GPU into the prompt cache. Not counted if the same tokens were already saved. |
| `prompt_cache_loads_total` | counter | Times a saved entry was copied back from the prompt cache into a slot's GPU KV cache. The entry is removed from the prompt cache when this happens. |
| `prompt_cache_loaded_tokens_total` | counter | Over all loads, how many tokens at the start of the new prompt matched the restored entry. This is the part of `prompt_tokens_cached_total` that came from CPU RAM rather than tokens already on the GPU (see the note below). |
| `prompt_cache_evictions_total` | counter | Entries deleted to stay under the size limit. The oldest entry is always deleted first. Entries removed because a newer, longer entry contains them are not counted. |

### Existing llama.cpp counters (for reference)

| Metric | Type | What it represents |
| --- | --- | --- |
| `prompt_tokens_total` | counter | Prompt tokens the server actually computed (cache misses). |
| `prompt_tokens_cached_total` | counter | Prompt tokens reused instead of computed (cache hits), from either the GPU KV cache or a restored prompt cache entry. |

### GPU hits vs CPU RAM hits

Both kinds count as cache hits, but they don't cost the same:

- **GPU hit:** the prefix is already in the slot's KV cache, so the server
  skips those tokens. Almost free.
- **CPU RAM hit:** the server first copies the whole saved entry from RAM into
  the slot's GPU KV cache, then skips the matching prefix. Much cheaper than
  recomputing, but the copy takes time that grows with the entry's size.

Over a time window, take the change (Δ) in each counter:

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

**Per-request counts.** Every completion response already reports its own
cached tokens: `usage.prompt_tokens_details.cached_tokens` (OpenAI-style) or
`timings.cache_n` (llama.cpp's field). Log it next to each request's latency
to see which requests hit.

## What is a slot?

A **slot** is one request lane inside llama-server. `PARALLEL=2` gives two
slots, so at most two requests run at the same time; the rest wait in a queue
(`requests_deferred`).

Each slot owns one sequence in the KV cache. That sequence holds the slot's
current tokens: the prompt plus the generated output. The cache is split evenly
between slots, so with `CTX_SIZE=4096` and `PARALLEL=2`, each slot can hold up
to 2048 tokens.

When a request finishes, the slot becomes idle, but its tokens **stay** in the
KV cache. If the next request on that slot starts with the same tokens (a
shared prefix, such as the same system prompt or the same conversation
history), the server skips recomputing them. That is a prefix-cache hit.

## Gauges vs counters

`prompt_tokens_total` and `prompt_tokens_cached_total` are **counters**: they
only go up, and you compare two readings to get the change in between.

The KV metrics are **gauges**: each reading is a snapshot of the cache when you
call `/metrics`. They go up and down. To study occupancy over a run, sample
them regularly (for example, once per second) and report the peak and average.

## How `kv_cache_tokens` is calculated

The server has one **slot** per parallel request (`PARALLEL=2` gives slots 0
and 1). Each slot stores its tokens in the KV cache as its own sequence, with
sequence id = slot id. A sequence covers positions `pos_min` to `pos_max`.

When you call `/metrics`, the server asks the KV cache, for each slot:

```
for each slot:
    pos_min = llama_memory_seq_pos_min(mem, slot.id)
    pos_max = llama_memory_seq_pos_max(mem, slot.id)
    if pos_max >= 0:            # -1 means the slot holds nothing
        kv_cache_tokens += pos_max - pos_min + 1

kv_cache_tokens_max = n_ctx
```

This reads the cache directly, so it counts what is actually stored, not what
the scheduler thinks is stored.

### Example

`CTX_SIZE=4096`, `PARALLEL=2`. Each slot can hold up to 4096 / 2 = 2048 tokens.

| Moment | Slot 0 | Slot 1 | `kv_cache_tokens` | `kv_cache_usage_ratio` |
| --- | --- | --- | --- | --- |
| Server just started | empty | empty | 0 | 0.00 |
| Request A: 600-token prompt + 100 generated | 700 | empty | 700 | 0.17 |
| A finishes (slot 0 is now idle) | 700 | empty | 700 | 0.17 |
| Request B: 1000-token prompt + 200 generated | 700 | 1200 | 1900 | 0.46 |
| Request C: short 50-token prompt, unrelated to A, goes to slot 0 | 50 + output | 1200 | about 1300 | 0.32 |

Two things to notice:

- **Usage stays up after a request finishes.** An idle slot keeps its tokens so
  that the next request with the same beginning (prefix) can reuse them. Those
  reused tokens are what `prompt_tokens_cached_total` counts.
- **Usage can drop when a new request arrives.** The server keeps only the part
  of the old tokens that matches the start of the new prompt and drops the
  rest. In row 5, C shares nothing with A, so A's 700 tokens are evicted.

## llama.cpp's default cache policy (our setup)

These are llama.cpp's defaults at our pinned commit; we don't change any of
them. The cache has two levels:

| Level | Where | Size | Counted in `kv_cache_tokens`? |
| --- | --- | --- | --- |
| KV cache | GPU memory | `CTX_SIZE` tokens, split between slots | Yes |
| Prompt cache | CPU RAM | up to 8192 MiB (`--cache-ram`) | No |

What happens when a request arrives:

1. **Pick a slot.** Among idle slots, choose the one whose stored tokens best
   match the start of the new prompt, if at least 10% of the prompt matches
   (`--slot-prompt-similarity 0.1`). Otherwise choose the least recently used
   idle slot. If no slot is idle, the request waits in the queue.
2. **Save the old contents to CPU RAM.** If the chosen slot was picked by
   "least recently used", or if less than half of its old tokens will survive,
   its old KV state is copied into the prompt cache before being overwritten.
   In addition, every time a request starts, each idle slot is copied into the
   prompt cache (unless an identical copy is already there). The idle slot
   keeps its tokens on the GPU too; this is only a backup.
3. **Restore from CPU RAM if that's better.** The server searches the prompt
   cache for a saved state that matches the new prompt better than what is in
   the slot. If it finds one, it copies that state back into the slot's KV
   cache.
4. **Reuse the prefix.** Inside the slot, keep the longest common prefix
   between the stored tokens and the new prompt, delete the rest, and compute
   only the new tokens.

Prompt cache eviction: when it exceeds its size limit, the server deletes the
**oldest saved entry first**, i.e. first in, first out, regardless of how often
an entry is used.

Two consequences for measurements:

- `prompt_tokens_cached_total` counts both kinds of hits: prefixes already in
  the GPU KV cache, and prefixes restored from CPU RAM in step 3. Use
  `prompt_cache_loaded_tokens_total` to separate them (see above).
- Data in the prompt cache is measured by `prompt_cache_tokens` and
  `prompt_cache_bytes`, not `kv_cache_tokens`. It is also part of the server's
  RSS.

Because we pass `--parallel 2` explicitly, each slot has its own fixed share of
the KV cache (no unified KV cache), and idle slots are not cleared between
requests. If the number of slots is left on auto, llama.cpp instead uses one
shared cache and moves idle slots to CPU RAM.

## Notes

- Patch: `inference_setup/llama-kv-metrics.patch`, tested at the llama.cpp
  commit in `llama_cpp_commit.txt`. It changes `tools/server/server-task.h`,
  `server-task.cpp` and `server-context.cpp`.
- If `grep kv_cache` or `grep prompt_cache` prints nothing, the running server
  is an older build. Rerun `install_everything.sh` and restart
  `start_server.sh`.
- `install_everything.sh` resets those three files in `llama.cpp/` before
  applying the patch, so any manual edits to them are discarded.
