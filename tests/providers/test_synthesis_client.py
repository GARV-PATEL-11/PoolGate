"""Provider-layer tests for SynthesisClient — audio endpoint mocking."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.synthesis_client import SynthesisClient, SynthesisResult
from exceptions.keys import APIKeyDisabledError
from exceptions.rate_limit import RateLimitExceededError


def _mock_audio_response(audio_bytes: bytes = b"audio_data") -> MagicMock:
    response = MagicMock()
    response.read.return_value = audio_bytes
    return response


def _fake_exc(status_code: int) -> Exception:
    err = Exception("sdk error")
    err.status_code = status_code  # type: ignore[attr-defined]
    return err


class RateLimitError(Exception):
    pass


@pytest.fixture
def client() -> SynthesisClient:
    return SynthesisClient()


# ---------------------------------------------------------------------------
# Sync synthesize
# ---------------------------------------------------------------------------

class TestSynthesizeSync:
    def test_returns_synthesis_result(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.return_value = _mock_audio_response(b"mp3_bytes")
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        result = client.synthesize(
            api_key="gsk_test",
            model="canopylabs/orpheus-v1-english",
            text="Hello world",
            voice="zoe",
            session_id="s1",
            api_key_id="key_0",
        )
        assert isinstance(result, SynthesisResult)
        assert result.audio == b"mp3_bytes"

    def test_voice_forwarded_to_sdk(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.return_value = _mock_audio_response()
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        client.synthesize(
            api_key="gsk_test",
            model="canopylabs/orpheus-v1-english",
            text="Hello",
            voice="aria",
            session_id="s1",
            api_key_id="key_0",
        )
        call_kwargs = mock_sdk.audio.speech.create.call_args.kwargs
        assert call_kwargs["voice"] == "aria"

    def test_response_format_forwarded(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.return_value = _mock_audio_response()
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        client.synthesize(
            api_key="gsk_test",
            model="canopylabs/orpheus-v1-english",
            text="Hello",
            voice="zoe",
            session_id="s1",
            api_key_id="key_0",
            response_format="wav",
        )
        call_kwargs = mock_sdk.audio.speech.create.call_args.kwargs
        assert call_kwargs["response_format"] == "wav"
        assert mock_sdk.audio.speech.create.call_args.kwargs["response_format"] == "wav"

    def test_speed_forwarded(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.return_value = _mock_audio_response()
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        client.synthesize(
            api_key="gsk_test",
            model="canopylabs/orpheus-v1-english",
            text="Hello",
            voice="zoe",
            session_id="s1",
            api_key_id="key_0",
            speed=1.5,
        )
        call_kwargs = mock_sdk.audio.speech.create.call_args.kwargs
        assert call_kwargs["speed"] == 1.5

    def test_result_voice_matches_request(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.return_value = _mock_audio_response()
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        result = client.synthesize(
            api_key="gsk_test",
            model="canopylabs/orpheus-v1-english",
            text="Hello",
            voice="zoe",
            session_id="s1",
            api_key_id="key_0",
        )
        assert result.voice == "zoe"

    def test_latency_is_non_negative(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.return_value = _mock_audio_response()
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        result = client.synthesize(
            api_key="gsk_test",
            model="canopylabs/orpheus-v1-english",
            text="Hello",
            voice="zoe",
            session_id="s1",
            api_key_id="key_0",
        )
        assert result.latency >= 0.0


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------

class TestSynthesisErrorMapping:
    def test_status_401_raises_api_key_disabled(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.side_effect = _fake_exc(401)
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        with pytest.raises(APIKeyDisabledError):
            client.synthesize(
                api_key="gsk_test",
                model="canopylabs/orpheus-v1-english",
                text="Hello",
                voice="zoe",
                session_id="s1",
                api_key_id="key_0",
            )

    def test_rate_limit_raises_rate_limit_exceeded(self, client, monkeypatch):
        exc = RateLimitError("rate limited")
        exc.response = None  # type: ignore[attr-defined]
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.side_effect = exc
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        with pytest.raises(RateLimitExceededError):
            client.synthesize(
                api_key="gsk_test",
                model="canopylabs/orpheus-v1-english",
                text="Hello",
                voice="zoe",
                session_id="s1",
                api_key_id="key_0",
            )


# ---------------------------------------------------------------------------
# Async synthesize
# ---------------------------------------------------------------------------

class TestAsyncSynthesis:
    @pytest.mark.asyncio
    async def test_async_synthesize_returns_result(self, client, monkeypatch):
        mock_response = MagicMock()
        mock_response.read.return_value = b"async_audio"
        mock_sdk = AsyncMock()
        mock_sdk.audio.speech.create = AsyncMock(return_value=mock_response)
        monkeypatch.setattr(client, "_async_sdk", lambda key: mock_sdk)

        result = await client.async_synthesize(
            api_key="gsk_test",
            model="canopylabs/orpheus-v1-english",
            text="Hello async",
            voice="zoe",
            session_id="s1",
            api_key_id="key_0",
        )
        assert isinstance(result, SynthesisResult)
        assert result.audio == b"async_audio"

    @pytest.mark.asyncio
    async def test_async_synthesize_auth_error_raises(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.audio.speech.create.side_effect = _fake_exc(403)
        monkeypatch.setattr(client, "_async_sdk", lambda key: mock_sdk)

        with pytest.raises(APIKeyDisabledError):
            await client.async_synthesize(
                api_key="gsk_test",
                model="canopylabs/orpheus-v1-english",
                text="Hello",
                voice="zoe",
                session_id="s1",
                api_key_id="key_0",
            )
