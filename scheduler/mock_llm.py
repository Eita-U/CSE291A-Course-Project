"""Small async backend for local integration without an inference server."""

import asyncio
from typing import Any


class MockLLMBackend:
    def __init__(self, delay_s: float = 0.01) -> None:
        self.delay_s = delay_s

    async def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(self.delay_s)
        # No fabricated token counts: a real backend supplies tokenizer usage.
        return {"response": f"Mock response to: {request['prompt']}"}
