# Notes: how llama-server caches prompts

Background for reading the metrics in [kv_cache_metrics.md](kv_cache_metrics.md).
Everything here describes llama.cpp's default behavior at the commit in
`llama_cpp_commit.txt`; we don't change any of it.

## What is a slot?

A **slot** is one request lane inside llama-server. `PARALLEL=2` gives two
slots, so at most two requests run at the same time; the rest wait in a queue
(`requests_deferred`).

Each slot owns one sequence in the KV cache. That sequence holds the slot's
current tokens: the prompt plus the generated output. Because we pass
`--parallel 2` explicitly, the cache is split evenly between slots: with
`CTX_SIZE=4096`, each slot can hold up to 2048 tokens.

A sequence is one linear stream of tokens, and each token attends to all the
tokens before it. So a slot can hold one conversation at a time, not several
unrelated requests side by side.

When a request finishes, the slot becomes idle, but its tokens **stay** in the
KV cache. If the next request on that slot starts with the same tokens (a
shared prefix, such as the same system prompt or the same conversation
history), the server skips recomputing them. That is a prefix-cache hit.

## Two cache levels

| Level | Where | Size | Measured by |
| --- | --- | --- | --- |
| KV cache | GPU memory | `CTX_SIZE` tokens, split between slots | `kv_cache_*` |
| Prompt cache | CPU RAM | up to 8192 MiB (`--cache-ram`) | `prompt_cache_*` |

- **GPU hit:** the prefix is already in the slot's KV cache, so the server
  skips those tokens. Almost free.
- **CPU RAM hit:** the server first copies the whole saved entry from RAM into
  the slot's GPU KV cache, then skips the matching prefix. Much cheaper than
  recomputing, but the copy takes time that grows with the entry's size.

Both kinds are counted in `prompt_tokens_cached_total`.

## What happens when a request arrives

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
   cache and removes it from the prompt cache.
4. **Reuse the prefix.** Inside the slot, keep the longest common prefix
   between the stored tokens and the new prompt, delete the rest, and compute
   only the new tokens.

**Prompt cache eviction:** when it exceeds its size limit, the server deletes
the **oldest saved entry first** (first in, first out), regardless of how often
an entry is used.

If the number of slots is left on auto, llama.cpp instead uses one shared KV
cache for all slots and moves idle slots to CPU RAM. That doesn't apply to our
`start_server.sh`.

## Example: GPU KV cache over time

`CTX_SIZE=4096`, `PARALLEL=2`. Each slot can hold up to 2048 tokens.

| Moment | Slot 0 | Slot 1 | `kv_cache_tokens` | `kv_cache_usage_ratio` |
| --- | --- | --- | --- | --- |
| Server just started | empty | empty | 0 | 0.00 |
| Request A: 600-token prompt + 100 generated | 700 | empty | 700 | 0.17 |
| A finishes (slot 0 is now idle) | 700 | empty | 700 | 0.17 |
| Request B: 1000-token prompt + 200 generated | 700 | 1200 | 1900 | 0.46 |
| Request C: short 50-token prompt, unrelated to A, goes to slot 0 | 50 + output | 1200 | about 1300 | 0.32 |

- **Usage stays up after a request finishes.** An idle slot keeps its tokens so
  the next request with the same prefix can reuse them.
- **Usage can drop when a new request arrives.** The server keeps only the part
  of the old tokens that matches the start of the new prompt and drops the
  rest. In the last row, C shares nothing with A, so A's 700 tokens are removed
  from the GPU (a backup was saved to CPU RAM in step 2).

## Example: seeing the CPU RAM cache work

Sending the same prompt twice uses only the GPU, so all `prompt_cache_*`
metrics stay 0. To exercise the RAM cache, send three different, long prompts
A, B, C and then A again, one after another:

| Step | What the server does | Metrics |
| --- | --- | --- |
| A | Runs on slot 0 | none of these change |
| B | Runs on slot 1; idle slot 0 (A) is backed up to RAM | saves = 1 |
| C | Overwrites slot 0, which held A (already in RAM); idle slot 1 (B) is backed up | saves = 2 |
| A again | Neither slot holds A, so the least recently used slot (slot 1, B) is chosen. A is copied back from RAM into it. | loads = 1, loaded_tokens ≈ size of A's prompt |

Use prompts of 100+ words that differ from the start. Short prompts share the
chat template's opening tokens, which can push them over the 10% similarity
threshold.
