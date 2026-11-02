"""Run every Task 3 microbenchmark and write traces plus graphs.

From the repository root:

    python3 -m examples.run_microbenchmarks
    python3 -m examples.run_microbenchmarks --quick
    python3 -m examples.run_microbenchmarks --out results/microbenchmarks

    # Real Qwen (llama.cpp already running, or SSH-tunneled to 127.0.0.1:8080):
    python3 -m examples.run_microbenchmarks --backend qwen --quick
    python3 -m examples.run_microbenchmarks --backend qwen --out results/qwen

The scheduler, agent runner, and Qwen HTTP adapter are not modified. Default
runs use WorkloadMockLLM. --backend qwen injects QwenHTTPBackend as the
existing backend_factory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from workload.benchmarks import BenchConfig, BackendFactory, quick_config, run_microbenchmarks
from workload.plots import generate_plots

DEFAULT_QWEN_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_QWEN_TIMEOUT_S = 180.0


class _PromptSuffixBackend:
    """Microbenchmark-only wrapper; does not change QwenHTTPBackend."""

    def __init__(self, inner: Any, suffix: str) -> None:
        self._inner = inner
        self._suffix = suffix

    async def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        req = dict(request)
        prompt = str(req.get("prompt") or "")
        if self._suffix and self._suffix not in prompt:
            req["prompt"] = f"{prompt} {self._suffix}".strip()
        return await self._inner.generate(req)

    async def aclose(self) -> None:
        closer = getattr(self._inner, "aclose", None)
        if callable(closer):
            result = closer()
            if asyncio.iscoroutine(result):
                await result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FIFO microbenchmarks and plot results")
    parser.add_argument("--out", type=Path, default=Path("results/microbenchmarks"))
    parser.add_argument("--quick", action="store_true", help="Smaller token budgets and fewer tasks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=("mock", "qwen"),
        default="mock",
        help="mock: local WorkloadMockLLM. qwen: HTTP QwenHTTPBackend (server must already be up).",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_QWEN_ENDPOINT,
        help="Exact generate URL for --backend qwen (OpenAI chat completions by default).",
    )
    parser.add_argument(
        "--api-style",
        choices=("openai", "llama_cpp"),
        default="openai",
        help="Must match the running llama.cpp route (see inference_setup/inference_api.md).",
    )
    parser.add_argument("--model", default=None, help="Optional model id; auto-detected from /v1/models when omitted")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Scheduler request timeout in seconds. Default 60 (mock) or 180 (qwen).",
    )
    parser.add_argument(
        "--backend-timeout",
        type=float,
        default=DEFAULT_QWEN_TIMEOUT_S,
        help="HTTP client timeout for --backend qwen.",
    )
    parser.add_argument(
        "--allow-think",
        action="store_true",
        help="Do not append /no_think to Qwen prompts (Qwen3 reasoning tokens count toward max_tokens).",
    )
    return parser.parse_args(argv)


def _base_url(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    return urlunsplit((parts.scheme, parts.netloc, "", "", "")).rstrip("/")


async def _discover_qwen_model(endpoint: str, timeout_s: float) -> str | None:
    import httpx

    base = _base_url(endpoint)
    async with httpx.AsyncClient(timeout=min(timeout_s, 15.0)) as client:
        health = await client.get(f"{base}/health")
        health.raise_for_status()
        try:
            models_response = await client.get(f"{base}/v1/models")
            models_response.raise_for_status()
        except httpx.HTTPError:
            return None
        payload = models_response.json()
        models = payload.get("data", []) if isinstance(payload, dict) else []
        if models and isinstance(models[0].get("id"), str):
            return models[0]["id"]
    return None


def qwen_backend_factory(
    *,
    endpoint: str,
    api_style: str,
    model: str | None,
    timeout_s: float,
    disable_think: bool,
    client: Any = None,
) -> BackendFactory:
    from scheduler.qwen_backend import QwenHTTPBackend

    def factory() -> Any:
        backend: Any = QwenHTTPBackend(
            endpoint,
            api_style=api_style,  # type: ignore[arg-type]
            model=model,
            timeout_s=timeout_s,
            client=client,
        )
        if disable_think:
            backend = _PromptSuffixBackend(backend, "/no_think")
        return backend

    return factory


def build_config(args: argparse.Namespace) -> BenchConfig:
    overrides: dict[str, Any] = {
        "seed": args.seed,
        "backend_label": args.backend,
    }
    if args.timeout is not None:
        overrides["request_timeout_s"] = args.timeout
    elif args.backend == "qwen":
        overrides["request_timeout_s"] = DEFAULT_QWEN_TIMEOUT_S
    if args.quick:
        return quick_config(args.out, **overrides)
    return BenchConfig(out_dir=args.out, **overrides)


async def _main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    factory: BackendFactory | None = None
    if args.backend == "qwen":
        model = args.model
        try:
            discovered = await _discover_qwen_model(args.endpoint, args.backend_timeout)
        except Exception as exc:
            print(
                "Qwen inference server is not reachable at "
                f"{_base_url(args.endpoint)} ({type(exc).__name__}: {exc}).\n"
                "Start it with bash inference_setup/start_server.sh, or SSH-tunnel "
                "port 8080 as in inference_setup/README.md, then rerun with --backend qwen.",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        if model is None:
            model = discovered
        print(
            f"Using QwenHTTPBackend endpoint={args.endpoint} "
            f"api_style={args.api_style} model={model!r}",
            file=sys.stderr,
        )
        factory = qwen_backend_factory(
            endpoint=args.endpoint,
            api_style=args.api_style,
            model=model,
            timeout_s=args.backend_timeout,
            disable_think=not args.allow_think,
        )
    config = build_config(args)
    report = await run_microbenchmarks(config, backend_factory=factory)
    paths = generate_plots(report, args.out / "graphs")
    summary = {
        "out_dir": str(args.out.resolve()),
        "backend": args.backend,
        "endpoint": args.endpoint if args.backend == "qwen" else None,
        "suites": [
            {
                "name": suite["name"],
                "concurrency": suite["concurrency"],
                "n_requests": suite["metrics"]["n_requests"],
                "n_tasks": suite["metrics"]["n_tasks"],
                "throughput_requests_per_s": suite["metrics"]["throughput_requests_per_s"],
                "request_total_ms_p95": suite["metrics"]["request_total_ms"]["p95"],
                "task_completion_ms_p95": suite["metrics"]["task_completion_ms"]["p95"],
                "peak_rss_mib": suite["peak_rss_bytes"] / (1024 * 1024),
            }
            for suite in report["suites"]
        ],
        "graphs": [str(path) for path in paths],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
