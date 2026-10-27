"""Offline HTTP adapter tests; no AWS access required."""

import asyncio
import unittest

import httpx

from scheduler import FIFOScheduler
from scheduler.qwen_backend import QwenHTTPBackend


class QwenHTTPBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_response_through_fifo(self):
        seen = []

        def responder(request):
            body = __import__("json").loads(request.content)
            seen.append(body)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "Answer"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
            backend = QwenHTTPBackend("http://example.test/v1/chat/completions", client=client)
            async with FIFOScheduler(backend, concurrency=2) as scheduler:
                result = await scheduler.submit({
                    "request_id": "test-1", "task_id": "agent-1", "step_id": 0,
                    "prompt": "Hello", "generation_settings": {"max_tokens": 8},
                })
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["response"], "Answer")
        self.assertEqual(result["token_usage"], {"prompt_tokens": 4, "completion_tokens": 1})
        self.assertEqual(seen[0]["messages"][0]["content"], "Hello")
        self.assertEqual(seen[0]["max_tokens"], 8)

    async def test_llamacpp_prompt_and_usage(self):
        def responder(req):
            body = __import__("json").loads(req.content)
            self.assertEqual(body["n_predict"], 16)
            self.assertEqual(body["prompt"], "Hello")
            return httpx.Response(200, json={
                "content": "Native answer", "tokens_evaluated": 5,
                "tokens_predicted": 3,
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
            backend = QwenHTTPBackend("http://example.test/completion", api_style="llama_cpp", client=client)
            answer = await backend.generate({"prompt": "Hello", "generation_settings": {"max_tokens": 16}})
        self.assertEqual(answer, {"response": "Native answer", "token_usage": {"prompt_tokens": 5, "completion_tokens": 3}})

    async def test_http_failure_propagates_to_scheduler(self):
        transport = httpx.MockTransport(lambda req: httpx.Response(503, json={"error": "down"}))
        async with httpx.AsyncClient(transport=transport) as client:
            backend = QwenHTTPBackend("http://example.test/v1/chat/completions", client=client)
            async with FIFOScheduler(backend) as scheduler:
                result = await scheduler.submit({
                    "request_id": "test-err", "task_id": "agent-1", "step_id": 0,
                    "prompt": "Hello",
                })
        self.assertEqual(result["status"], "failed")
        self.assertIn("HTTP 503", result["error"])

    async def test_missing_usage_not_fabricated(self):
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json={
            "choices": [{"message": {"content": "Okay"}}],
        }))
        async with httpx.AsyncClient(transport=transport) as client:
            backend = QwenHTTPBackend("http://example.test/v1/chat/completions", client=client)
            result = await backend.generate({"prompt": "Hi"})
        self.assertEqual(result["token_usage"], {})

    async def test_invalid_settings_dont_make_http_request(self):
        called = False
        def responder(req):
            nonlocal called
            called = True
            return httpx.Response(200, json={})
        async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
            backend = QwenHTTPBackend("http://example.test/completion", api_style="llama_cpp", client=client)
            with self.assertRaises(ValueError):
                await backend.generate({"prompt": "Hi", "generation_settings": {"stream": True}})
        self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
