"""Unit tests for clients/synthesis_client.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.synthesis_client import SynthesisClient, SynthesisResult

_FAKE_AUDIO = b"\xff\xfb\x90\x00" * 256
_MODEL = "canopylabs/orpheus-v1-english"
_VOICE = "zoe"


def _mock_speech_response(audio: bytes = _FAKE_AUDIO):
    response = MagicMock()
    response.read.return_value = audio
    return response


@pytest.fixture
def client():
    return SynthesisClient()


class TestSynthesize:

    def test_returns_synthesis_result(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.return_value = _mock_speech_response()
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        result = client.synthesize(
            api_key="gsk_test",
            model=_MODEL,
            text="Hello, world.",
            voice=_VOICE,
            session_id="s1",
            api_key_id="key_0",
        )
        assert isinstance(result, SynthesisResult)
        assert result.audio == _FAKE_AUDIO

    def test_audio_is_bytes(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.return_value = _mock_speech_response()
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        result = client.synthesize(
            api_key="gsk_test",
            model=_MODEL,
            text="test",
            voice=_VOICE,
            session_id="s1",
            api_key_id="key_0",
        )
        assert isinstance(result.audio, bytes)

    def test_voice_is_forwarded(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.return_value = _mock_speech_response()
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        client.synthesize(
            api_key="gsk_test",
            model=_MODEL,
            text="hi",
            voice=_VOICE,
            session_id="s1",
            api_key_id="key_0",
        )
        call_kwargs = mock_sdk.audio.speech.create.call_args.kwargs
        assert call_kwargs["voice"] == _VOICE

    def test_result_has_correct_model_and_voice(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.return_value = _mock_speech_response()
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        result = client.synthesize(
            api_key="gsk_test",
            model=_MODEL,
            text="hi",
            voice=_VOICE,
            session_id="s1",
            api_key_id="key_0",
        )
        assert result.model == _MODEL
        assert result.voice == _VOICE

    def test_default_response_format_is_mp3(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.return_value = _mock_speech_response()
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        result = client.synthesize(
            api_key="gsk_test",
            model=_MODEL,
            text="hi",
            voice=_VOICE,
            session_id="s1",
            api_key_id="key_0",
        )
        assert result.response_format == "mp3"

    def test_custom_response_format(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.return_value = _mock_speech_response()
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        result = client.synthesize(
            api_key="gsk_test",
            model=_MODEL,
            text="hi",
            voice=_VOICE,
            session_id="s1",
            api_key_id="key_0",
            response_format="wav",
        )
        assert result.response_format == "wav"

    def test_latency_is_non_negative(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.speech.create.return_value = _mock_speech_response()
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        result = client.synthesize(
            api_key="gsk_test",
            model=_MODEL,
            text="hi",
            voice=_VOICE,
            session_id="s1",
            api_key_id="key_0",
        )
        assert result.latency >= 0.0


class TestAsyncSynthesize:

    @pytest.mark.asyncio
    async def test_async_synthesize_returns_result(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.audio.speech.create = AsyncMock(return_value=_mock_speech_response())
        monkeypatch.setattr(client, "_async_sdk", lambda api_key: mock_sdk)

        result = await client.async_synthesize(
            api_key="gsk_test",
            model=_MODEL,
            text="Hello async world",
            voice=_VOICE,
            session_id="s1",
            api_key_id="key_0",
        )
        assert isinstance(result, SynthesisResult)
        assert result.audio == _FAKE_AUDIO
        assert result.voice == _VOICE

    @pytest.mark.asyncio
    async def test_async_synthesize_calls_speech_endpoint(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.audio.speech.create = AsyncMock(return_value=_mock_speech_response())
        monkeypatch.setattr(client, "_async_sdk", lambda api_key: mock_sdk)

        await client.async_synthesize(
            api_key="gsk_test",
            model=_MODEL,
            text="test",
            voice=_VOICE,
            session_id="s1",
            api_key_id="key_0",
        )
        assert mock_sdk.audio.speech.create.called
