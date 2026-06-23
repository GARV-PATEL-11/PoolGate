"""
Integration tests for services/provider_service.py.

All Groq SDK calls are mocked via monkeypatch on _sync_sdk / _async_sdk —
no real network calls. Tests verify the full request lifecycle: key
acquisition, client dispatch, tracking updates, and error propagation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from exceptions.keys import NoAvailableAPIKeyError
from exceptions.response import RetryExhaustedError
from schemas.ops import HealthStatus
from schemas.runtime import RequestConfig
from services.provider_service import GroqService

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mock_completion(text: str, prompt_tokens: int = 5, completion_tokens: int = 3):
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=text), finish_reason="stop")]
    completion.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return completion


def _mock_stream_chunks(texts: list[str]):
    """Build a mock stream context manager yielding one chunk per text."""
    chunks = []
    for i, text in enumerate(texts):
        chunk = MagicMock(x_groq=None)
        chunk.choices = [MagicMock(delta=MagicMock(content=text))]
        chunks.append(chunk)
    # Final chunk with usage
    final = MagicMock()
    final.choices = [MagicMock(delta=MagicMock(content=""))]
    final.x_groq = MagicMock()
    final.x_groq.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chunks.append(final)

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=iter(chunks))
    mock_stream.__exit__ = MagicMock(return_value=False)
    return mock_stream


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("TOTAL_GROQ_KEYS", "2")
    monkeypatch.setenv("GROQ_API_KEY_01", "gsk_key_1")
    monkeypatch.setenv("GROQ_API_KEY_02", "gsk_key_2")

    return GroqService()


@pytest.fixture
def mock_sdk(service, monkeypatch):
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = _mock_completion("mocked answer")
    monkeypatch.setattr(service._chat_client, "_sync_sdk", lambda api_key: sdk)
    return sdk


# ---------------------------------------------------------------------------
# invoke() lifecycle
# ---------------------------------------------------------------------------


class TestInvokeLifecycle:

    def test_invoke_returns_response_with_text(self, service, mock_sdk):
        response = service.invoke("What is 1+1?", model="llama-3.3-70b-versatile")
        assert response.text == "mocked answer"
        assert response.model == "llama-3.3-70b-versatile"
        assert response.latency > 0

    def test_invoke_accumulates_global_stats(self, service, mock_sdk):
        service.invoke("Hello", model="llama-3.3-70b-versatile")
        service.invoke("World", model="llama-3.3-70b-versatile")
        stats = service.get_global_stats()
        assert stats["total_requests"] >= 2
        assert stats["successful_requests"] >= 2

    def test_invoke_updates_key_pool_rpm(self, service, mock_sdk):
        service.invoke("Hello", model="llama-3.3-70b-versatile")
        pool = service.get_key_pool_status()
        used = sum(k["requests_per_minute"] for k in pool)
        assert used >= 1

    def test_invoke_with_system_prompt(self, service, mock_sdk):
        response = service.invoke(
            "Tell me a joke.",
            model="llama-3.3-70b-versatile",
            system="You are a comedian.",
        )
        assert response.text == "mocked answer"
        # Verify system message was included in SDK call
        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles


# ---------------------------------------------------------------------------
# chat() lifecycle
# ---------------------------------------------------------------------------


class TestChatLifecycle:

    def test_chat_with_multi_turn_messages(self, service, mock_sdk):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        response = service.chat(messages, model="llama-3.3-70b-versatile")
        assert response.text == "mocked answer"


# ---------------------------------------------------------------------------
# stream() lifecycle
# ---------------------------------------------------------------------------


class TestStreamLifecycle:

    def test_stream_yields_chunks_in_order(self, service, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_stream_chunks(["Hello", " world"])
        monkeypatch.setattr(service._chat_client, "_sync_sdk", lambda api_key: mock_sdk)

        chunks = list(service.stream("Say hello", model="llama-3.3-70b-versatile"))
        # Filter out the empty final chunk
        non_empty = [c for c in chunks if c]
        assert non_empty == ["Hello", " world"]

    def test_stream_records_usage_tracking(self, service, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_stream_chunks(["answer"])
        monkeypatch.setattr(service._chat_client, "_sync_sdk", lambda api_key: mock_sdk)

        list(service.stream("Question", model="llama-3.3-70b-versatile"))
        stats = service.get_global_stats()
        assert stats["successful_requests"] >= 1


# ---------------------------------------------------------------------------
# structured() lifecycle
# ---------------------------------------------------------------------------


class TestStructuredLifecycle:

    def test_structured_returns_pydantic_instance(self, service, monkeypatch):
        class Answer(BaseModel):
            value: int

        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion('{"value": 42}')
        monkeypatch.setattr(service._structured_client, "_sync_sdk", lambda api_key: mock_sdk)

        result = service.structured("What is 6*7?", Answer, model="llama-3.3-70b-versatile")
        assert isinstance(result, Answer)
        assert result.value == 42

    def test_structured_raises_on_invalid_json(self, service, monkeypatch):
        class Strict(BaseModel):
            value: int

        from exceptions.output import StructuredOutputError

        mock_sdk = MagicMock()
        # Return invalid JSON and keep returning it (for all retries)
        mock_sdk.chat.completions.create.return_value = _mock_completion("not valid json at all")
        monkeypatch.setattr(service._structured_client, "_sync_sdk", lambda api_key: mock_sdk)

        with pytest.raises(StructuredOutputError):
            service.structured("prompt", Strict, model="llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# batch() lifecycle
# ---------------------------------------------------------------------------


class TestBatchLifecycle:

    def test_batch_returns_summary_with_all_results(self, service, monkeypatch):
        mock_async_sdk = AsyncMock()
        mock_async_sdk.chat.completions.create = AsyncMock(return_value=_mock_completion("answer"))
        monkeypatch.setattr(service._chat_client, "_async_sdk", lambda api_key: mock_async_sdk)

        prompts = ["Q1", "Q2", "Q3"]
        summary = asyncio.run(service.batch(prompts, model="llama-3.3-70b-versatile"))
        assert summary.total == 3
        assert summary.succeeded == 3
        assert summary.failed == 0
        assert len(summary.results) == 3
        assert all(r.success for r in summary.results)

    def test_batch_results_sorted_by_index(self, service, monkeypatch):
        mock_async_sdk = AsyncMock()
        mock_async_sdk.chat.completions.create = AsyncMock(return_value=_mock_completion("ok"))
        monkeypatch.setattr(service._chat_client, "_async_sdk", lambda api_key: mock_async_sdk)

        summary = asyncio.run(service.batch(["a", "b", "c"], model="llama-3.3-70b-versatile"))
        assert [r.index for r in summary.results] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Key rotation on failure
# ---------------------------------------------------------------------------


class TestKeyRotation:

    def test_second_key_used_when_first_fails(self, service, monkeypatch):
        call_count = [0]
        success_sdk = MagicMock()
        success_sdk.chat.completions.create.return_value = _mock_completion("fallback answer")

        def factory(api_key: str) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                failing = MagicMock()
                failing.chat.completions.create.side_effect = RuntimeError("first key fail")
                return failing
            return success_sdk

        monkeypatch.setattr(service._chat_client, "_sync_sdk", factory)

        cfg = RequestConfig(retries=1)
        response = service.invoke("Hi", model="llama-3.3-70b-versatile", config=cfg)
        assert response.text == "fallback answer"
        assert call_count[0] == 2

    def test_all_keys_fail_raises_retry_exhausted(self, service, monkeypatch):
        def factory(api_key: str) -> MagicMock:
            sdk = MagicMock()
            sdk.chat.completions.create.side_effect = RuntimeError("always fails")
            return sdk

        monkeypatch.setattr(service._chat_client, "_sync_sdk", factory)

        cfg = RequestConfig(retries=1)
        with pytest.raises(RetryExhaustedError):
            service.invoke("Hi", model="llama-3.3-70b-versatile", config=cfg)

    def test_no_available_key_raises_immediately(self, service, monkeypatch):
        # Disable all keys directly via the scheduler's internal key list
        for k in service._scheduler._keys:
            k.mark_disabled()

        with pytest.raises(NoAvailableAPIKeyError):
            service.invoke("Hi", model="llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# health() and stats
# ---------------------------------------------------------------------------


class TestHealthAndStats:

    def test_health_returns_health_status(self, service, mock_sdk):
        health = service.health()
        assert isinstance(health, HealthStatus)
        assert health.status in ("healthy", "degraded", "unhealthy")
        assert health.active_keys >= 1
        assert health.uptime_seconds >= 0

    def test_health_shows_healthy_for_active_keys(self, service, mock_sdk):
        health = service.health()
        assert health.status == "healthy"

    def test_global_stats_accumulate_after_calls(self, service, mock_sdk):
        before = service.get_global_stats()["total_requests"]
        service.invoke("Q", model="llama-3.3-70b-versatile")
        service.invoke("R", model="llama-3.3-70b-versatile")
        after = service.get_global_stats()["total_requests"]
        assert after == before + 2

    def test_global_stats_has_all_required_keys(self, service):
        stats = service.get_global_stats()
        expected = {
            "total_requests",
            "successful_requests",
            "failed_requests",
            "total_retries",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "success_rate",
        }
        assert expected.issubset(stats.keys())


# ---------------------------------------------------------------------------
# flush_tracking()
# ---------------------------------------------------------------------------


class TestFlushTracking:

    def test_flush_tracking_noop_without_persistence(self, service, mock_sdk):
        service.invoke("Hello", model="llama-3.3-70b-versatile")
        # Should not raise even without persistence configured
        service.flush_tracking()

    def test_flush_tracking_delegates_to_persistence(
        self,
        service,
        mock_sdk,
        tmp_path,
        monkeypatch,
    ):
        from services.persistence_service import PersistenceService

        db_path = tmp_path / "usage.db"
        persistence = PersistenceService.sqlite(db_path)

        # Reconstruct with persistence
        monkeypatch.setenv("TOTAL_GROQ_KEYS", "2")

        monkeypatch.setenv("GROQ_API_KEY_01", "gsk_key_1")
        monkeypatch.setenv("GROQ_API_KEY_02", "gsk_key_2")
        svc_with_persistence = GroqService(persistence=persistence)
        mock = MagicMock()
        mock.chat.completions.create.return_value = _mock_completion("ok")
        monkeypatch.setattr(svc_with_persistence._chat_client, "_sync_sdk", lambda api_key: mock)

        svc_with_persistence.invoke("Hello", model="llama-3.3-70b-versatile")
        svc_with_persistence.flush_tracking()

        # Reload and verify something was persisted
        reloaded = PersistenceService.sqlite(db_path)
        data = reloaded.load_all()
        assert len(data) >= 1  # at least today's bucket written
