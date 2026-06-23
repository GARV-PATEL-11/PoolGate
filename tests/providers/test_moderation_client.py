"""Provider-layer tests for ModerationClient — error mapping and argument forwarding."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.moderation_client import ModerationClient, ModerationResult
from exceptions.keys import APIKeyDisabledError
from exceptions.rate_limit import RateLimitExceededError
from schemas.runtime import RequestConfig


def _mock_moderation_completion(label: str = "SAFE", prompt_tokens: int = 8) -> MagicMock:
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = label

    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=1,
        total_tokens=prompt_tokens + 1,
    )
    return completion


def _fake_exc(status_code: int) -> Exception:
    err = Exception("sdk error")
    err.status_code = status_code  # type: ignore[attr-defined]
    return err


class RateLimitError(Exception):
    pass


@pytest.fixture
def client() -> ModerationClient:
    return ModerationClient()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestModerationErrorMapping:

    def test_status_401_raises_api_key_disabled(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.side_effect = _fake_exc(401)
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        with pytest.raises(APIKeyDisabledError):
            client.moderate(
                api_key="gsk_test",
                model="meta-llama/llama-prompt-guard-2-86m",
                text="test",
                config=RequestConfig(),
                session_id="s1",
                api_key_id="key_0",
            )

    def test_status_403_raises_api_key_disabled(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.side_effect = _fake_exc(403)
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        with pytest.raises(APIKeyDisabledError):
            client.moderate(
                api_key="gsk_test",
                model="meta-llama/llama-prompt-guard-2-86m",
                text="test",
                config=RequestConfig(),
                session_id="s1",
                api_key_id="key_0",
            )

    def test_rate_limit_error_raises_rate_limit_exceeded(self, client, monkeypatch):
        exc = RateLimitError("rate limited")
        exc.response = None  # type: ignore[attr-defined]
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.side_effect = exc
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        with pytest.raises(RateLimitExceededError):
            client.moderate(
                api_key="gsk_test",
                model="meta-llama/llama-prompt-guard-2-86m",
                text="harm this",
                config=RequestConfig(),
                session_id="s1",
                api_key_id="key_0",
            )


# ---------------------------------------------------------------------------
# Async moderate
# ---------------------------------------------------------------------------


class TestAsyncModerationProvider:

    @pytest.mark.asyncio
    async def test_async_moderate_returns_moderation_result(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_moderation_completion("SAFE"),
        )
        monkeypatch.setattr(client, "_async_sdk", lambda key: mock_sdk)

        result = await client.async_moderate(
            api_key="gsk_test",
            model="meta-llama/llama-prompt-guard-2-86m",
            text="Hello world",
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
        )
        assert isinstance(result, ModerationResult)
        assert result.label == "SAFE"

    @pytest.mark.asyncio
    async def test_async_moderate_auth_error_raises(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create.side_effect = _fake_exc(401)
        monkeypatch.setattr(client, "_async_sdk", lambda key: mock_sdk)

        with pytest.raises(APIKeyDisabledError):
            await client.async_moderate(
                api_key="gsk_test",
                model="meta-llama/llama-prompt-guard-2-86m",
                text="test",
                config=RequestConfig(),
                session_id="s1",
                api_key_id="key_0",
            )


# ---------------------------------------------------------------------------
# Argument forwarding
# ---------------------------------------------------------------------------


class TestModerationArgumentForwarding:

    def test_model_forwarded_to_sdk(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_moderation_completion("SAFE")
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        model = "meta-llama/llama-prompt-guard-2-86m"
        client.moderate(
            api_key="gsk_test",
            model=model,
            text="some content",
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
        )
        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == model

    def test_session_id_in_result(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_moderation_completion("SAFE")
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        result = client.moderate(
            api_key="gsk_test",
            model="meta-llama/llama-prompt-guard-2-86m",
            text="test",
            config=RequestConfig(),
            session_id="my-session",
            api_key_id="key_0",
        )
        assert result.session_id == "my-session"

    def test_request_id_propagated(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_moderation_completion("SAFE")
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        result = client.moderate(
            api_key="gsk_test",
            model="meta-llama/llama-prompt-guard-2-86m",
            text="test",
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
            request_id="my-rid",
        )
        assert result.request_id == "my-rid"

    def test_temperature_forced_to_zero(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_moderation_completion("SAFE")
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        client.moderate(
            api_key="gsk_test",
            model="meta-llama/llama-prompt-guard-2-86m",
            text="test",
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
        )
        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0
