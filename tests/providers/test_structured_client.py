"""Provider-layer tests for StructuredClient — JSON mode and schema-constrained output."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.structured_client import StructuredClient
from exceptions.keys import APIKeyDisabledError
from exceptions.rate_limit import RateLimitExceededError
from schemas.runtime import RequestConfig


def _mock_json_completion(json_text: str = '{"value": 42}') -> MagicMock:
	choice = MagicMock()
	choice.finish_reason = "stop"
	choice.message.content = json_text

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


_MESSAGES = [{"role": "user", "content": "Give me a JSON object with value 42"}]


@pytest.fixture
def client() -> StructuredClient:
	return StructuredClient()


# ---------------------------------------------------------------------------
# Sync invoke_structured
# ---------------------------------------------------------------------------

class TestInvokeStructured:

	def test_returns_groq_response_with_json_text(self, client, monkeypatch):
		mock_sdk = MagicMock()
		mock_sdk.chat.completions.create.return_value = _mock_json_completion('{"value": 42}')
		monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

		response = client.invoke_structured(
			api_key="gsk_test",
			model="llama-3.3-70b-versatile",
			messages=_MESSAGES,
			config=RequestConfig(),
			session_id="s1",
			api_key_id="key_0",
			)
		import json

		data = json.loads(response.text)
		assert data["value"] == 42

	def test_json_object_mode_by_default(self, client, monkeypatch):
		mock_sdk = MagicMock()
		mock_sdk.chat.completions.create.return_value = _mock_json_completion("{}")
		monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

		client.invoke_structured(
			api_key="gsk_test",
			model="llama-3.3-70b-versatile",
			messages=_MESSAGES,
			config=RequestConfig(),
			session_id="s1",
			api_key_id="key_0",
			json_schema=None,
			)
		call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
		assert call_kwargs["response_format"]["type"] == "json_object"

	def test_json_schema_forwarded_to_sdk(self, client, monkeypatch):
		mock_sdk = MagicMock()
		mock_sdk.chat.completions.create.return_value = _mock_json_completion("{}")
		monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

		schema = {"title": "Answer", "type": "object", "properties": {"value": {"type": "integer"}}}
		client.invoke_structured(
			api_key="gsk_test",
			model="llama-3.3-70b-versatile",
			messages=_MESSAGES,
			config=RequestConfig(),
			session_id="s1",
			api_key_id="key_0",
			json_schema=schema,
			)
		call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
		assert call_kwargs["response_format"]["type"] == "json_schema"

	def test_latency_non_negative(self, client, monkeypatch):
		mock_sdk = MagicMock()
		mock_sdk.chat.completions.create.return_value = _mock_json_completion("{}")
		monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

		response = client.invoke_structured(
			api_key="gsk_test",
			model="llama-3.3-70b-versatile",
			messages=_MESSAGES,
			config=RequestConfig(),
			session_id="s1",
			api_key_id="key_0",
			)
		assert response.latency >= 0.0


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

class TestStructuredErrorMapping:

	def test_status_401_raises_api_key_disabled(self, client, monkeypatch):
		mock_sdk = MagicMock()
		mock_sdk.chat.completions.create.side_effect = _fake_exc(401)
		monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

		with pytest.raises(APIKeyDisabledError):
			client.invoke_structured(
				api_key="gsk_test",
				model="llama-3.3-70b-versatile",
				messages=_MESSAGES,
				config=RequestConfig(),
				session_id="s1",
				api_key_id="key_0",
				)

	def test_rate_limit_raises_rate_limit_exceeded(self, client, monkeypatch):
		exc = RateLimitError("rate limited")
		exc.response = None  # type: ignore[attr-defined]
		mock_sdk = MagicMock()
		mock_sdk.chat.completions.create.side_effect = exc
		monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

		with pytest.raises(RateLimitExceededError):
			client.invoke_structured(
				api_key="gsk_test",
				model="llama-3.3-70b-versatile",
				messages=_MESSAGES,
				config=RequestConfig(),
				session_id="s1",
				api_key_id="key_0",
				)


# ---------------------------------------------------------------------------
# Async invoke_structured
# ---------------------------------------------------------------------------

class TestAsyncInvokeStructured:

	@pytest.mark.asyncio
	async def test_async_returns_response_with_json_text(self, client, monkeypatch):
		mock_sdk = AsyncMock()
		mock_sdk.chat.completions.create = AsyncMock(
			return_value=_mock_json_completion('{"answer": "yes"}'),
			)
		monkeypatch.setattr(client, "_async_sdk", lambda key: mock_sdk)

		response = await client.async_invoke_structured(
			api_key="gsk_test",
			model="llama-3.3-70b-versatile",
			messages=_MESSAGES,
			config=RequestConfig(),
			session_id="s1",
			api_key_id="key_0",
			)
		import json

		assert json.loads(response.text)["answer"] == "yes"

	@pytest.mark.asyncio
	async def test_async_auth_error_raises(self, client, monkeypatch):
		mock_sdk = AsyncMock()
		mock_sdk.chat.completions.create.side_effect = _fake_exc(403)
		monkeypatch.setattr(client, "_async_sdk", lambda key: mock_sdk)

		with pytest.raises(APIKeyDisabledError):
			await client.async_invoke_structured(
				api_key="gsk_test",
				model="llama-3.3-70b-versatile",
				messages=_MESSAGES,
				config=RequestConfig(),
				session_id="s1",
				api_key_id="key_0",
				)
