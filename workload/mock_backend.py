"""Local backend that honors max_tokens so workload tests need no GPU."""

from __future__ import annotations

import asyncio
from typing import Any


class WorkloadMockLLM:
    """Sleeps and emits a response sized from generation_settings.max_tokens."""

    def __init__(
        self,
        delay_s: float = 0.01,
        per_token_delay_s: float = 0.0,
        per_prompt_token_delay_s: float = 0.0,
    ) -> None:
        self.delay_s = delay_s
        self.per_token_delay_s = per_token_delay_s
        self.per_prompt_token_delay_s = per_prompt_token_delay_s

    async def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        settings = request.get("generation_settings") or {}
        max_tokens = int(settings.get("max_tokens") or 8)
        max_tokens = max(1, max_tokens)
        prompt = str(request.get("prompt") or "")
        prompt_tokens = max(1, len(prompt.split()))
        await asyncio.sleep(
            self.delay_s
            + self.per_prompt_token_delay_s * prompt_tokens
            + self.per_token_delay_s * max_tokens
        )
        response = " ".join(f"tok{i}" for i in range(max_tokens))
        return {
            "response": response,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max_tokens,
            },
        }
