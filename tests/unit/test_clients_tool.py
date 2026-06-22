"""Unit tests for clients/tool_client.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.tool_client import ToolClient
from schemas.runtime import FinishReason, RequestConfig

_SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


def _mock_tool_completion(finish_reason: str = "tool_calls", content: str = ""):
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "get_weather"
    tool_call.function.arguments = '{"location": "Paris"}'

    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = [tool_call] if finish_reason == "tool_calls" else []

    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return completion


@pytest.fixture
def client():
    return ToolClient()


class TestInvokeTools:
    def test_returns_groq_response_with_tool_calls_finish_reason(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_tool_completion("tool_calls")
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        response = client.invoke_tools(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
            tools=_SAMPLE_TOOLS,
            tool_choice="auto",
        )
        assert response.finish_reason == FinishReason.TOOL_CALLS

    def test_tool_choice_none_returns_stop(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_tool_completion(
            "stop", content="I can't use tools."
        )
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        response = client.invoke_tools(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
            tools=_SAMPLE_TOOLS,
            tool_choice="none",
        )
        assert response.finish_reason == FinishReason.STOP

    def test_sdk_is_called_with_tools_argument(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_tool_completion()
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        client.invoke_tools(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "weather?"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
            tools=_SAMPLE_TOOLS,
        )
        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] == _SAMPLE_TOOLS

    def test_usage_is_populated(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_tool_completion()
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        response = client.invoke_tools(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
            tools=_SAMPLE_TOOLS,
        )
        assert response.usage.prompt_tokens == 10


class TestAsyncInvokeTools:
    @pytest.mark.asyncio
    async def test_async_returns_groq_response(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_tool_completion("tool_calls")
        )
        monkeypatch.setattr(client, "_async_sdk", lambda api_key: mock_sdk)

        response = await client.async_invoke_tools(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "weather?"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
            tools=_SAMPLE_TOOLS,
        )
        assert response.finish_reason == FinishReason.TOOL_CALLS
