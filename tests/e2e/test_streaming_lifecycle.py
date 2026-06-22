"""E2E tests for streaming lifecycle — stream() and async_stream() via mocked SDK."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from exceptions.response import RetryExhaustedError
from schemas.runtime import RequestConfig
from services.provider_service import GroqService
from tests.e2e.conftest import _mock_stream_chunks


def _set_groq_keys(monkeypatch, keys: list[str]) -> None:
    monkeypatch.setenv("TOTAL_GROQ_KEYS", str(len(keys)))
    for i, key in enumerate(keys, start=1):
        monkeypatch.setenv(f"GROQ_API_KEY_{i:02d}", key)


@pytest.fixture
def service(monkeypatch) -> GroqService:
    _set_groq_keys(monkeypatch, ["gsk_stream_key_1", "gsk_stream_key_2"])
    return GroqService()


def _setup_stream_sdk(service: GroqService, monkeypatch, texts: list[str]) -> MagicMock:
    mock_sdk = MagicMock()
    mock_sdk.chat.completions.create.return_value = _mock_stream_chunks(texts)
    monkeypatch.setattr(service._chat_client, "_sync_sdk", lambda key: mock_sdk)
    return mock_sdk


# ---------------------------------------------------------------------------
# Sync stream
# ---------------------------------------------------------------------------

class TestStreamLifecycle:
    def test_stream_yields_chunks_in_order(self, service, monkeypatch):
        _setup_stream_sdk(service, monkeypatch, ["Hello", " world", "!"])
        chunks = list(service.stream("Say hello world!", model="llama-3.3-70b-versatile"))
        non_empty = [c for c in chunks if c]
        assert non_empty == ["Hello", " world", "!"]

    def test_stream_records_tracking(self, service, monkeypatch):
        _setup_stream_sdk(service, monkeypatch, ["answer"])
        before = service.get_global_stats()["successful_requests"]
        list(service.stream("Q", model="llama-3.3-70b-versatile"))
        after = service.get_global_stats()["successful_requests"]
        assert after == before + 1

    def test_stream_updates_key_rpm(self, service, monkeypatch):
        _setup_stream_sdk(service, monkeypatch, ["token"])
        list(service.stream("Prompt", model="llama-3.3-70b-versatile"))
        pool = service.get_key_pool_status()
        used = sum(k["requests_per_minute"] for k in pool)
        assert used >= 1

    def test_stream_failure_before_first_chunk_retried(self, service, monkeypatch):
        call_count = [0]
        success_sdk = MagicMock()
        success_sdk.chat.completions.create.return_value = _mock_stream_chunks(["ok"])

        def factory(key: str) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                failing = MagicMock()
                failing.chat.completions.create.side_effect = RuntimeError("first attempt fails")
                return failing
            return success_sdk

        monkeypatch.setattr(service._chat_client, "_sync_sdk", factory)

        cfg = RequestConfig(stream=True, retries=1)
        chunks = list(service.stream("Q", model="llama-3.3-70b-versatile", config=cfg))
        non_empty = [c for c in chunks if c]
        assert non_empty == ["ok"]
        assert call_count[0] == 2

    def test_stream_all_attempts_fail_raises_retry_exhausted(self, service, monkeypatch):
        def factory(key: str) -> MagicMock:
            sdk = MagicMock()
            sdk.chat.completions.create.side_effect = RuntimeError("always fails")
            return sdk

        monkeypatch.setattr(service._chat_client, "_sync_sdk", factory)
        cfg = RequestConfig(stream=True, retries=1)

        with pytest.raises(RetryExhaustedError):
            list(service.stream("Q", model="llama-3.3-70b-versatile", config=cfg))

    def test_stream_with_messages_param(self, service, monkeypatch):
        _setup_stream_sdk(service, monkeypatch, ["response"])
        messages = [{"role": "user", "content": "hello"}]
        chunks = list(service.stream(messages=messages, model="llama-3.3-70b-versatile"))
        assert any(c for c in chunks if c)


# ---------------------------------------------------------------------------
# Async stream
# ---------------------------------------------------------------------------

def _make_async_stream_chunks(texts: list[str]):
    chunks = []
    for text in texts:
        chunk = MagicMock(x_groq=None)
        chunk.choices = [MagicMock(delta=MagicMock(content=text))]
        chunks.append(chunk)
    final = MagicMock()
    final.choices = [MagicMock(delta=MagicMock(content=""))]
    final.x_groq = MagicMock()
    final.x_groq.usage = MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    chunks.append(final)
    return chunks


class TestAsyncStreamLifecycle:
    @pytest.mark.asyncio
    async def test_async_stream_yields_chunks(self, service, monkeypatch):
        async def fake_async_stream(*args, **kwargs):
            yield "async"
            yield " response"

        monkeypatch.setattr(
            service._chat_client,
            "async_stream",
            fake_async_stream,
        )

        collected = []
        async for chunk in service.async_stream("Q", model="llama-3.3-70b-versatile"):
            collected.append(chunk)

        non_empty = [c for c in collected if c]
        assert non_empty == ["async", " response"]

    @pytest.mark.asyncio
    async def test_async_stream_records_tracking(self, service, monkeypatch):
        chunks_data = _make_async_stream_chunks(["data"])

        async def fake_async_stream(*args, **kwargs):
            for chunk in chunks_data:
                yield chunk

        monkeypatch.setattr(service._chat_client, "async_stream", fake_async_stream)

        before = service.get_global_stats()["successful_requests"]
        async for _ in service.async_stream("Q", model="llama-3.3-70b-versatile"):
            pass
        after = service.get_global_stats()["successful_requests"]
        assert after == before + 1
