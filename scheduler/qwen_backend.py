"""Async HTTP adapter connecting the FIFO scheduler to a Qwen inference service.

Supported protocols:
- OpenAI-compatible ``/v1/chat/completions`` (e.g., vLLM/llama.cpp OpenAI route)
- llama.cpp native ``/completion``

The endpoint and protocol must match the inference service Isabella deployed.
This module deliberately does not assume either one is running.

Install: python -m pip install httpx
"""

from __future__ import annotations

from typing import Any, Literal

import httpx


class QwenHTTPBackend:
    """HTTP backend implementing ``async generate(request) -> dict``.

    Pass an exact endpoint URL, such as http://HOST:PORT/v1/chat/completions
    or http://HOST:PORT/completion. The backend does not start the model server.

    The owner must call ``await backend.aclose()`` after scheduler shutdown.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        api_style: Literal["openai", "llama_cpp"] = "openai",
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("endpoint must be an http:// or https:// URL")
        if api_style not in ("openai", "llama_cpp"):
            raise ValueError("api_style must be 'openai' or 'llama_cpp'")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.endpoint = endpoint
        self.api_style = api_style
        self.model = model
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        """Transform the scheduler's prompt/settings to a server request.

        The FIFO scheduler validates top-level request fields. We validate
        protocol-specific settings here, and don't fabricate token counts.
        """
        prompt = request["prompt"]
        settings = dict(request.get("generation_settings", {}))
        if settings.get("stream") is True:
            raise ValueError("stream=True is unsupported; use a non-streaming request")
        settings.pop("stream", None)

        if self.api_style == "openai":
            body: dict[str, Any] = {
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            if self.model is not None:
                body["model"] = self.model
            body.update(settings)
        else:
            # llama.cpp native /completion names this setting n_predict.
            if "max_tokens" in settings:
                if "n_predict" in settings:
                    raise ValueError("Use either max_tokens or n_predict, not both")
                settings["n_predict"] = settings.pop("max_tokens")
            body = {"prompt": prompt, "stream": False, **settings}

        try:
            response = await self._client.post(
                self.endpoint, json=body, headers=self._headers,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TimeoutError("Qwen HTTP request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Qwen API returned HTTP {exc.response.status_code}") from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Qwen API connection error: {type(exc).__name__}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError("Qwen API response was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("Qwen API JSON response must be an object")

        if self.api_style == "openai":
            try:
                text = payload["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("OpenAI-compatible response missing choices[0].message.content") from exc
            usage_raw = payload.get("usage", {})
            if usage_raw is None:
                usage_raw = {}
            if not isinstance(usage_raw, dict):
                raise ValueError("Qwen token usage must be a dictionary")
            usage = {
                key: usage_raw[key]
                for key in ("prompt_tokens", "completion_tokens")
                if key in usage_raw and type(usage_raw[key]) is int and usage_raw[key] >= 0
            }
        else:
            text = payload.get("content")
            usage = {}
            # llama.cpp native /completion exposes these counters on many versions.
            for out_key, in_key in (("prompt_tokens", "tokens_evaluated"),
                                    ("completion_tokens", "tokens_predicted")):
                count = payload.get(in_key)
                if type(count) is int and count >= 0:
                    usage[out_key] = count
            timings = payload.get("timings")
            if isinstance(timings, dict):
                for out_key, in_key in (("prompt_tokens", "prompt_n"),
                                        ("completion_tokens", "predicted_n")):
                    count = timings.get(in_key)
                    if out_key not in usage and type(count) is int and count >= 0:
                        usage[out_key] = count

        if not isinstance(text, str):
            raise ValueError("Qwen API response must contain generated text as a string")
        return {"response": text, "token_usage": usage}

    async def aclose(self) -> None:
        """Close the internally-owned HTTP connection pool."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "QwenHTTPBackend":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()
