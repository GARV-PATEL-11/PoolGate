"""Provider-layer tests for TranscriptionClient — audio transcription/translation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from poolgate.capabilities.transcription import TranscriptionCapability as TranscriptionClient, TranscriptionResult
from poolgate.exceptions.keys import APIKeyDisabledError
from poolgate.exceptions.rate_limit import RateLimitExceededError


def _mock_transcription_result(text: str = "transcribed text") -> MagicMock:
    result = MagicMock()
    result.text = text
    return result


def _fake_exc(status_code: int) -> Exception:
    err = Exception("sdk error")
    err.status_code = status_code  # type: ignore[attr-defined]
    return err


class RateLimitError(Exception):
    pass


_FAKE_AUDIO = b"fake_audio_bytes"


@pytest.fixture
def client() -> TranscriptionClient:
    return TranscriptionClient()


# ---------------------------------------------------------------------------
# Sync transcribe
# ---------------------------------------------------------------------------


class TestTranscribeSync:

    def test_returns_transcription_result(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.transcriptions.create.return_value = _mock_transcription_result("hello")
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        result = client.transcribe(
            api_key="gsk_test",
            model="whisper-large-v3",
            audio_file=_FAKE_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello"
        assert result.task == "transcribe"

    def test_language_forwarded(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.transcriptions.create.return_value = _mock_transcription_result()
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        client.transcribe(
            api_key="gsk_test",
            model="whisper-large-v3",
            audio_file=_FAKE_AUDIO,
            session_id="s1",
            api_key_id="key_0",
            language="fr",
        )
        call_kwargs = mock_sdk.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["language"] == "fr"

    def test_latency_non_negative(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.transcriptions.create.return_value = _mock_transcription_result()
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        result = client.transcribe(
            api_key="gsk_test",
            model="whisper-large-v3",
            audio_file=_FAKE_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert result.latency >= 0.0


# ---------------------------------------------------------------------------
# Sync translate
# ---------------------------------------------------------------------------


class TestTranslateSync:

    def test_returns_transcription_result_with_translate_task(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.translations.create.return_value = _mock_transcription_result("translated")
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        result = client.translate(
            api_key="gsk_test",
            model="whisper-large-v3",
            audio_file=_FAKE_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert isinstance(result, TranscriptionResult)
        assert result.text == "translated"
        assert result.task == "translate"
        assert result.language is None  # translate always outputs English

    def test_translate_uses_translations_endpoint(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.translations.create.return_value = _mock_transcription_result()
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        client.translate(
            api_key="gsk_test",
            model="whisper-large-v3",
            audio_file=_FAKE_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert mock_sdk.audio.translations.create.called
        assert not mock_sdk.audio.transcriptions.create.called


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestTranscriptionErrorMapping:

    def test_status_401_raises_api_key_disabled(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.transcriptions.create.side_effect = _fake_exc(401)
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        with pytest.raises(APIKeyDisabledError):
            client.transcribe(
                api_key="gsk_test",
                model="whisper-large-v3",
                audio_file=_FAKE_AUDIO,
                session_id="s1",
                api_key_id="key_0",
            )

    def test_rate_limit_raises_rate_limit_exceeded(self, client, monkeypatch):
        exc = RateLimitError("rate limited")
        exc.response = None  # type: ignore[attr-defined]
        mock_sdk = MagicMock()
        mock_sdk.audio.transcriptions.create.side_effect = exc
        monkeypatch.setattr(client, "_sync_sdk", lambda key: mock_sdk)

        with pytest.raises(RateLimitExceededError):
            client.transcribe(
                api_key="gsk_test",
                model="whisper-large-v3",
                audio_file=_FAKE_AUDIO,
                session_id="s1",
                api_key_id="key_0",
            )


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------


class TestAsyncTranscription:

    @pytest.mark.asyncio
    async def test_async_transcribe_returns_result(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.audio.transcriptions.create = AsyncMock(
            return_value=_mock_transcription_result("async text"),
        )
        monkeypatch.setattr(client, "_async_sdk", lambda key: mock_sdk)

        result = await client.async_transcribe(
            api_key="gsk_test",
            model="whisper-large-v3",
            audio_file=_FAKE_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert isinstance(result, TranscriptionResult)
        assert result.text == "async text"

    @pytest.mark.asyncio
    async def test_async_translate_returns_result(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.audio.translations.create = AsyncMock(
            return_value=_mock_transcription_result("async translated"),
        )
        monkeypatch.setattr(client, "_async_sdk", lambda key: mock_sdk)

        result = await client.async_translate(
            api_key="gsk_test",
            model="whisper-large-v3",
            audio_file=_FAKE_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert result.text == "async translated"
        assert result.task == "translate"

    @pytest.mark.asyncio
    async def test_async_transcribe_auth_error_raises(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.audio.transcriptions.create.side_effect = _fake_exc(403)
        monkeypatch.setattr(client, "_async_sdk", lambda key: mock_sdk)

        with pytest.raises(APIKeyDisabledError):
            await client.async_transcribe(
                api_key="gsk_test",
                model="whisper-large-v3",
                audio_file=_FAKE_AUDIO,
                session_id="s1",
                api_key_id="key_0",
            )
