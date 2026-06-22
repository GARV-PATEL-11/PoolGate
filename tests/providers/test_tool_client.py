"""Provider-layer tests for ToolClient — error mapping and argument forwarding."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.tool_client import ToolClient
from exceptions.keys import APIKeyDisabledError
from exceptions.rate_limit import RateLimitExceededError
from schemas.runtime import FinishReason, RequestConfig


_SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]

_SAMPLE_MESSAGES = [{"role": "user", "content": "What's the weather?"}]


def _mock_tool_completion(finish_reason: str = "tool_calls") -> MagicMock:
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = ""
    choice.message.tool_calls = []

    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return completion


def _fake_exc(status_code: int) -> Exception:
    err = Exception("sdk error")
    err.status_code = status_code  # type: ignore[attr-defined]
    return err


class RateLimitError(Exception):
    pass


@pytest.fixture
def client() -> ToolClient:
    return ToolClient()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

class TestToolClientErrorMapping:
    def test_status_401_raises_api_key_disabled(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.side_effect = _fake_exc(401)
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        with pytest.raises(APIKeyDisabledError):
            client.invoke_tools(
                api_key="gsk_test",
                model="llama-3.3-70b-versatile",
                messages=_SAMPLE_MESSAGES,
                config=RequestConfig(),
                session_id="s1",
                api_key_id="key_0",
                tools=_SAMPLE_TOOLS,
            )

    def test_rate_limit_error_raises_rate_limit_exceeded(self, client, monkeypatch):
        exc = RateLimitError("rate limited")
        exc.response = None  # type: ignore[attr-defined]
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.side_effect = exc
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        with pytest.raises(RateLimitExceededError):
            client.invoke_tools(
                api_key="gsk_test",
                model="llama-3.3-70b-versatile",
                messages=_SAMPLE_MESSAGES,
                config=RequestConfig(),
                session_id="s1",
                api_key_id="key_0",
                tools=_SAMPLE_TOOLS,
            )


# ---------------------------------------------------------------------------
# tool_choice variants
# ---------------------------------------------------------------------------

class TestToolClientChoiceVariants:
    def test_tool_choice_auto_forwarded(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_tool_completion()
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        client.invoke_tools(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=_SAMPLE_MESSAGES,
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
            tools=_SAMPLE_TOOLS,
            tool_choice="auto",
        )
        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == "auto"

    def test_tool_choice_none_forwarded(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_tool_completion("stop")
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        client.invoke_tools(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=_SAMPLE_MESSAGES,
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
            tools=_SAMPLE_TOOLS,
            tool_choice="none",
        )
        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == "none"

    def test_tool_choice_dict_forwarded(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_tool_completion()
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        choice_dict = {"type": "function", "function": {"name": "get_weather"}}
        client.invoke_tools(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=_SAMPLE_MESSAGES,
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
            tools=_SAMPLE_TOOLS,
            tool_choice=choice_dict,
        )
        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == choice_dict


# ---------------------------------------------------------------------------
# Async invoke_tools
# ---------------------------------------------------------------------------

class TestAsyncToolClient:
    @pytest.mark.asyncio
    async def test_async_invoke_tools_returns_response(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_tool_completion("tool_calls")
        )
        monkeypatch.setattr(client, "_async_sdk", lambda key: mock_sdk)

        response = await client.async_invoke_tools(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=_SAMPLE_MESSAGES,
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
            tools=_SAMPLE_TOOLS,
        )
        assert response.finish_reason == FinishReason.TOOL_CALLS

    @pytest.mark.asyncio
    async def test_async_auth_error_raises(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create.side_effect = _fake_exc(401)
        monkeypatch.setattr(client, "_async_sdk", lambda key: mock_sdk)

        with pytest.raises(APIKeyDisabledError):
            await client.async_invoke_tools(
                api_key="gsk_test",
                model="llama-3.3-70b-versatile",
                messages=_SAMPLE_MESSAGES,
                config=RequestConfig(),
                session_id="s1",
                api_key_id="key_0",
                tools=_SAMPLE_TOOLS,
            )
