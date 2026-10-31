"""Manual end-to-end Qwen smoke test against Isabella's llama.cpp server.

Run from the CSE291A project root after SSH port forwarding is active:
    python -m examples.real_qwen_smoke

Requires: scheduler/qwen_backend.py, httpx.
This test makes *real* inference requests, so avoid running during benchmarks.
"""

import asyncio
import os

import httpx

from scheduler import FIFOScheduler
from scheduler.qwen_backend import QwenHTTPBackend

BASE_URL = os.getenv("QWEN_BASE_URL", "http://127.0.0.1:8080").rstrip("/")


async def main():
    # Fetch server-reported model identifier, as in inference_setup/smoke_test.py.
    async with httpx.AsyncClient(timeout=15) as client:
        health = await client.get(f"{BASE_URL}/health")
        health.raise_for_status()
        models_response = await client.get(f"{BASE_URL}/v1/models")
        models_response.raise_for_status()
        models = models_response.json().get("data", [])
        if not models or not isinstance(models[0].get("id"), str):
            raise RuntimeError("No model ID returned by /v1/models")
        model_id = models[0]["id"]
    print(f"Server reachable. Model: {model_id}")

    async with QwenHTTPBackend(
        endpoint=f"{BASE_URL}/v1/chat/completions",
        api_style="openai",
        model=model_id,
        timeout_s=180,
    ) as backend:
        # First verify one request traverses the FIFO -> adapter -> Qwen route.
        async with FIFOScheduler(backend, concurrency=1, request_timeout_s=180) as scheduler:
            single = await scheduler.submit({
                "request_id": "real-single-1", "task_id": "smoke", "step_id": 0,
                "prompt": "Reply in one short sentence: what is machine learning? /no_think",
                "generation_settings": {"max_tokens": 256, "temperature": 0},
            })
        print("Single request:", {k: single[k] for k in ("status", "response", "token_usage", "error")})
        if single["status"] != "success" or not single["response"]:
            raise RuntimeError("Single real-Qwen request did not return nonempty text")

        # Small live integration check; not a performance benchmark.
        async with FIFOScheduler(backend, concurrency=2, request_timeout_s=180) as scheduler:
            results = await asyncio.gather(*[
                scheduler.submit({
                    "request_id": f"real-concurrent-{i}", "task_id": "smoke", "step_id": i,
                    "prompt": f"Reply in one sentence: what is {topic}? /no_think",
                    "generation_settings": {"max_tokens": 256, "temperature": 0},
                })
                for i, topic in enumerate(("an algorithm", "a neural network"), 1)
            ])
        for res in results:
            print(res["request_id"], res["status"],
                  "queue_ms=", round(res["timing_metadata"]["queue_time_ms"], 1),
                  "api_ms=", round(res["timing_metadata"]["api_time_ms"], 1),
                  "error=", res["error"])
        if any(res["status"] != "success" or not res["response"] for res in results):
            raise RuntimeError("At least one concurrent real-Qwen request failed")

    print("PASS: FIFO -> QwenHTTPBackend -> real Qwen (single + concurrent)")


if __name__ == "__main__":
    asyncio.run(main())
