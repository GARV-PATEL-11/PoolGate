"""E2E tests for tool-calling lifecycle — invoke_tools() and async_invoke_tools()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from schemas.runtime import FinishReason
from services.provider_service import GroqService


def _set_groq_keys(monkeypatch, keys: list[str]) -> None:
    monkeypatch.setenv("TOTAL_GROQ_KEYS", str(len(keys)))
    for i, key in enumerate(keys, start=1):
        monkeypatch.setenv(f"GROQ_API_KEY_{i:02d}", key)


def _mock_tool_completion(finish_reason: str = "tool_calls") -> MagicMock:
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "get_weather"
    tool_call.function.arguments = '{"location": "Paris"}'

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = ""
    choice.message.tool_calls = [tool_call] if finish_reason == "tool_calls" else []

    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return completion


_SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
]

_MESSAGES = [{"role": "user", "content": "What's the weather in Paris?"}]


@pytest.fixture
def service(monkeypatch) -> GroqService:
    _set_groq_keys(monkeypatch, ["gsk_tools_key_1", "gsk_tools_key_2"])
    return GroqService()


# ---------------------------------------------------------------------------
# Sync invoke_tools
# ---------------------------------------------------------------------------


class TestToolCallingLifecycle:

    def test_invoke_tools_returns_response(self, service, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_tool_completion("tool_calls")
        monkeypatch.setattr(service._tool_client, "_sync_sdk", lambda key: mock_sdk)

        response = service.invoke_tools(
            messages=_MESSAGES,
            tools=_SAMPLE_TOOLS,
            model="llama-3.3-70b-versatile",
        )
        assert response.finish_reason == FinishReason.TOOL_CALLS

    def test_invoke_tools_records_tracking(self, service, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_tool_completion("tool_calls")
        monkeypatch.setattr(service._tool_client, "_sync_sdk", lambda key: mock_sdk)

        before = service.get_global_stats()["successful_requests"]
        service.invoke_tools(
            messages=_MESSAGES,
            tools=_SAMPLE_TOOLS,
            model="llama-3.3-70b-versatile",
        )
        after = service.get_global_stats()["successful_requests"]
        assert after == before + 1

    def test_invoke_tools_has_model_in_response(self, service, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_tool_completion()
        monkeypatch.setattr(service._tool_client, "_sync_sdk", lambda key: mock_sdk)

        response = service.invoke_tools(
            messages=_MESSAGES,
            tools=_SAMPLE_TOOLS,
            model="llama-3.3-70b-versatile",
        )
        assert response.model == "llama-3.3-70b-versatile"

    def test_invoke_tools_with_tool_choice_none(self, service, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_tool_completion("stop")
        monkeypatch.setattr(service._tool_client, "_sync_sdk", lambda key: mock_sdk)

        response = service.invoke_tools(
            messages=_MESSAGES,
            tools=_SAMPLE_TOOLS,
            model="llama-3.3-70b-versatile",
            tool_choice="none",
        )
        assert response.finish_reason == FinishReason.STOP

    def test_invoke_tools_updates_key_rpm(self, service, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_tool_completion()
        monkeypatch.setattr(service._tool_client, "_sync_sdk", lambda key: mock_sdk)

        service.invoke_tools(
            messages=_MESSAGES,
            tools=_SAMPLE_TOOLS,
            model="llama-3.3-70b-versatile",
        )
        pool = service.get_key_pool_status()
        used = sum(k["requests_per_minute"] for k in pool)
        assert used >= 1


# ---------------------------------------------------------------------------
# Async invoke_tools
# ---------------------------------------------------------------------------


class TestAsyncToolCallingLifecycle:

    @pytest.mark.asyncio
    async def test_async_invoke_tools_returns_response(self, service, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_tool_completion("tool_calls"),
        )
        monkeypatch.setattr(service._tool_client, "_async_sdk", lambda key: mock_sdk)

        response = await service.async_invoke_tools(
            messages=_MESSAGES,
            tools=_SAMPLE_TOOLS,
            model="llama-3.3-70b-versatile",
        )
        assert response.finish_reason == FinishReason.TOOL_CALLS

    @pytest.mark.asyncio
    async def test_async_invoke_tools_records_tracking(self, service, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_tool_completion(),
        )
        monkeypatch.setattr(service._tool_client, "_async_sdk", lambda key: mock_sdk)

        before = service.get_global_stats()["successful_requests"]
        await service.async_invoke_tools(
            messages=_MESSAGES,
            tools=_SAMPLE_TOOLS,
            model="llama-3.3-70b-versatile",
        )
        after = service.get_global_stats()["successful_requests"]
        assert after == before + 1
