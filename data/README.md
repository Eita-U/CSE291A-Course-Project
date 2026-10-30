# Task 4 trace data

Real agent traces for evaluating the FIFO baseline. Synthetic microbenchmarks are maintained separately in `workload/`.

## Source

- Dataset: [DiscoPosse/lmcache-agentic-traces](https://huggingface.co/datasets/DiscoPosse/lmcache-agentic-traces)
- License: CC BY 4.0
- Downloaded: 2026-10-25
- Revision: `9e1de874521be873b2c92621049ecb836b536257`

## Processing

Run [data_process.ipynb](data_process.ipynb) with `data/` as the working directory. Dependencies: `pandas`, `pyarrow`, `numpy`, `tqdm`, and `transformers`.

1. Read the five `raw/train-*.parquet` files in filename order and concatenate their rows.
2. Add a one-based `step` counter within each `session_id`, preserving row order.
3. Count `input_tokens` using the `Qwen/Qwen3-1.7B` tokenizer at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Apply its chat template with `add_generation_prompt=True` and `enable_thinking=True`, including template tokens in the count.
4. Keep a complete session only if **every step** satisfies `input_tokens + output_length <= 32768`.
5. Save to `processed/trace_32k.parquet` using PyArrow, without the DataFrame index. Keep raw files unchanged.

This excludes oversized sessions; it does not truncate messages or remove individual steps from retained sessions. Retained conversations preserve their original history and prefix growth. Results therefore describe the selected subset of sessions.

The notebook does not deduplicate, explicitly validate missing values, or sort session steps chronologically. Step order assumes the source rows are already in iteration order.

## Columns

One row represents one LLM iteration.

| Column | Description |
| --- | --- |
| `session_id` | Session identifier shared by all its iterations. |
| `model` | Model used in the original trace. |
| `input` | Cumulative OpenAI-format message list, including original previous responses and tool results. Preserved as nested data. |
| `output_length` | Completion tokens generated in the original iteration. |
| `pre_gap` | Seconds from the previous response completing to this request being sent; tool/processing/thinking time. First iteration: `0.0`. |
| `step` | One-based iteration number added by the notebook. |
| `input_tokens` | Input token count under the pinned Qwen chat template, including the generation prompt. |

## Replay notes

- Replay steps sequentially per session, waiting `pre_gap` after the previous response completes. Run independent sessions concurrently.
- Use each row's recorded `input` directly; do not append replay-generated responses to the next row's history.
- The scheduler schema and OpenAI-compatible HTTP backend accept structured `messages`; null optional message fields (`name`, `tool_calls`, `tool_call_id`) are omitted before sending.
- `output_length` can guide a generation limit, but does not guarantee the same generated length.
- The 32K filter combines Qwen input counts with the original model's completion counts. Match the serving template/thinking settings and configure sufficient context per server slot; the filter alone does not guarantee runtime fit.
