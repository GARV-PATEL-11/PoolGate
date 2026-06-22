"""Unit tests for clients/structured_client.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.structured_client import StructuredClient
from schemas.runtime import RequestConfig


def _mock_completion(text: str, prompt_tokens: int = 5, completion_tokens: int = 3):
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=text), finish_reason="stop")]
    completion.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return completion


@pytest.fixture
def client():
    return StructuredClient()


class TestInvokeStructured:
    def test_returns_groq_response_with_json_text(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion('{"answer": 42}')
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        response = client.invoke_structured(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "give me json"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
        )
        assert response.text == '{"answer": 42}'

    def test_usage_is_populated(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion(
            '{"x": 1}', prompt_tokens=7, completion_tokens=4
        )
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        response = client.invoke_structured(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "json"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
        )
        assert response.usage.prompt_tokens == 7
        assert response.usage.completion_tokens == 4

    def test_with_json_schema(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion('{"name": "Alice"}')
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        response = client.invoke_structured(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "name?"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
            json_schema=schema,
        )
        assert response.text == '{"name": "Alice"}'


class TestAsyncInvokeStructured:
    @pytest.mark.asyncio
    async def test_async_returns_groq_response(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_completion('{"result": "ok"}')
        )
        monkeypatch.setattr(client, "_async_sdk", lambda api_key: mock_sdk)

        response = await client.async_invoke_structured(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "json please"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
        )
        assert response.text == '{"result": "ok"}'

    @pytest.mark.asyncio
    async def test_async_usage_is_populated(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_completion("{}", prompt_tokens=3, completion_tokens=2)
        )
        monkeypatch.setattr(client, "_async_sdk", lambda api_key: mock_sdk)

        response = await client.async_invoke_structured(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "empty"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
        )
        assert response.usage.prompt_tokens == 3
