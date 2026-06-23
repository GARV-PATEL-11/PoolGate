"""Unit tests for clients/transcription_client.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.transcription_client import TranscriptionClient, TranscriptionResult


def _mock_transcription_result(text: str):
    result = MagicMock()
    result.text = text
    return result


@pytest.fixture
def client():
    return TranscriptionClient()


_AUDIO = b"\xff\xfb\x90\x00" * 100  # fake MP3-ish bytes
_MODEL = "whisper-large-v3"


class TestTranscribe:

    def test_returns_transcription_result(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.transcriptions.create.return_value = _mock_transcription_result(
            "Hello world",
        )
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        result = client.transcribe(
            api_key="gsk_test",
            model=_MODEL,
            audio_file=_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"

    def test_task_is_transcribe(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.transcriptions.create.return_value = _mock_transcription_result("hi")
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        result = client.transcribe(
            api_key="gsk_test",
            model=_MODEL,
            audio_file=_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert result.task == "transcribe"

    def test_language_is_passed_through(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.transcriptions.create.return_value = _mock_transcription_result("Bonjour")
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        result = client.transcribe(
            api_key="gsk_test",
            model=_MODEL,
            audio_file=_AUDIO,
            session_id="s1",
            api_key_id="key_0",
            language="fr",
        )
        assert result.language == "fr"

    def test_model_is_set_on_result(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.transcriptions.create.return_value = _mock_transcription_result("text")
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        result = client.transcribe(
            api_key="gsk_test",
            model=_MODEL,
            audio_file=_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert result.model == _MODEL

    def test_latency_is_non_negative(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.transcriptions.create.return_value = _mock_transcription_result("hi")
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        result = client.transcribe(
            api_key="gsk_test",
            model=_MODEL,
            audio_file=_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert result.latency >= 0.0


class TestTranslate:

    def test_returns_translation_result(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.translations.create.return_value = _mock_transcription_result(
            "Hello in English",
        )
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        result = client.translate(
            api_key="gsk_test",
            model=_MODEL,
            audio_file=_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert result.text == "Hello in English"
        assert result.task == "translate"
        assert result.language is None  # always English, no source language

    def test_uses_translations_endpoint(self, client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.audio.translations.create.return_value = _mock_transcription_result("text")
        monkeypatch.setattr(client, "_sync_sdk", lambda api_key: mock_sdk)

        client.translate(
            api_key="gsk_test",
            model=_MODEL,
            audio_file=_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert mock_sdk.audio.translations.create.called
        assert not mock_sdk.audio.transcriptions.create.called


class TestAsyncTranscribe:

    @pytest.mark.asyncio
    async def test_async_transcribe_returns_result(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.audio.transcriptions.create = AsyncMock(
            return_value=_mock_transcription_result("async transcription"),
        )
        monkeypatch.setattr(client, "_async_sdk", lambda api_key: mock_sdk)

        result = await client.async_transcribe(
            api_key="gsk_test",
            model=_MODEL,
            audio_file=_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert result.text == "async transcription"
        assert result.task == "transcribe"

    @pytest.mark.asyncio
    async def test_async_translate_returns_result(self, client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.audio.translations.create = AsyncMock(
            return_value=_mock_transcription_result("async translation"),
        )
        monkeypatch.setattr(client, "_async_sdk", lambda api_key: mock_sdk)

        result = await client.async_translate(
            api_key="gsk_test",
            model=_MODEL,
            audio_file=_AUDIO,
            session_id="s1",
            api_key_id="key_0",
        )
        assert result.text == "async translation"
        assert result.task == "translate"
        assert result.language is None
