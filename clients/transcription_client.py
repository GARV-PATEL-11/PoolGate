"""
transcription_client.py — TranscriptionClient

Speech-to-text transcription and optional English translation via the Groq
audio API (Whisper endpoint).

Supported models:
  whisper-large-v3
  whisper-large-v3-turbo

transcribe / async_transcribe
  Converts spoken audio to text in the original source language.

translate / async_translate
  Converts spoken audio to English regardless of the source language.
  Uses the /audio/translations endpoint — output is always English.

Public methods:
  transcribe()   — blocking transcription
  async_transcribe()  — async transcription
  translate()    — blocking translation-to-English
  async_translate()   — async translation-to-English
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import BinaryIO

from clients.base import _new_rid, BaseGroqClient
from clients.capabilities import TranscriptionCapability

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    """Structured result from a Whisper transcription or translation call."""

    text: str  # transcribed / translated text
    model: str
    latency: float
    session_id: str
    request_id: str
    api_key_id: str
    language: str | None = None  # BCP-47 source language tag; None for translate (output=EN)
    task: str = "transcribe"  # "transcribe" | "translate"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class TranscriptionClient(BaseGroqClient, TranscriptionCapability):
    """
    Stateless client for Whisper-based audio transcription and translation.

    audio_file
            Accepts any of the formats the Groq audio endpoint accepts:
              - an open binary file handle (e.g. open("clip.mp3", "rb"))
              - raw bytes
              - a (filename, bytes) tuple so the SDK can infer the MIME type

    response_format
            "text"    — plain UTF-8 string (default)
            "json"    — {"text": "..."} JSON object
            "verbose_json" — full word-level timestamps + segments

    Note: TokenUsage is not available on audio endpoints; latency is the
    only telemetry field populated beyond the text result.
    """

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def transcribe(
        self,
        api_key: str,
        model: str,
        audio_file: BinaryIO | bytes | tuple[str, bytes],
        session_id: str,
        api_key_id: str,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "text",
        temperature: float = 0.0,
        timeout: float | None = None,
        request_id: str | None = None,
    ) -> TranscriptionResult:
        """
        Blocking speech-to-text transcription in the source language.

        language — BCP-47 tag (e.g. "en", "ar", "fr").  Supply it when you
                           know the source language to skip Whisper's language detection
                           and improve accuracy.
        prompt   — optional text priming the model with expected vocabulary
                           or spelling conventions.
        """
        rid = _new_rid(request_id)
        client = self._sync_sdk(api_key)
        start = time.perf_counter()

        try:
            result = client.audio.transcriptions.create(
                file=audio_file,  # type: ignore[arg-type]
                model=model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)

        text = result.text if hasattr(result, "text") else str(result)
        return TranscriptionResult(
            text=text,
            model=model,
            latency=time.perf_counter() - start,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
            language=language,
            task="transcribe",
        )

    def translate(
        self,
        api_key: str,
        model: str,
        audio_file: BinaryIO | bytes | tuple[str, bytes],
        session_id: str,
        api_key_id: str,
        prompt: str | None = None,
        response_format: str = "text",
        temperature: float = 0.0,
        timeout: float | None = None,
        request_id: str | None = None,
    ) -> TranscriptionResult:
        """
        Blocking speech-to-English translation.

        No language parameter — output is always English.
        The source language is detected automatically by Whisper.
        """
        rid = _new_rid(request_id)
        client = self._sync_sdk(api_key)
        start = time.perf_counter()

        try:
            result = client.audio.translations.create(
                file=audio_file,  # type: ignore[arg-type]
                model=model,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)

        text = result.text if hasattr(result, "text") else str(result)
        return TranscriptionResult(
            text=text,
            model=model,
            latency=time.perf_counter() - start,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
            language=None,  # output is always English
            task="translate",
        )

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    async def async_transcribe(
        self,
        api_key: str,
        model: str,
        audio_file: BinaryIO | bytes | tuple[str, bytes],
        session_id: str,
        api_key_id: str,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "text",
        temperature: float = 0.0,
        timeout: float | None = None,
        request_id: str | None = None,
    ) -> TranscriptionResult:
        """
        Async speech-to-text transcription in the source language.

        Identical contract to transcribe() — uses the native async Groq SDK.
        """
        rid = _new_rid(request_id)
        client = self._async_sdk(api_key)
        start = time.perf_counter()

        try:
            result = await client.audio.transcriptions.create(
                file=audio_file,  # type: ignore[arg-type]
                model=model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)

        text = result.text if hasattr(result, "text") else str(result)
        return TranscriptionResult(
            text=text,
            model=model,
            latency=time.perf_counter() - start,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
            language=language,
            task="transcribe",
        )

    async def async_translate(
        self,
        api_key: str,
        model: str,
        audio_file: BinaryIO | bytes | tuple[str, bytes],
        session_id: str,
        api_key_id: str,
        prompt: str | None = None,
        response_format: str = "text",
        temperature: float = 0.0,
        timeout: float | None = None,
        request_id: str | None = None,
    ) -> TranscriptionResult:
        """
        Async speech-to-English translation.

        Identical contract to translate() — uses the native async Groq SDK.
        """
        rid = _new_rid(request_id)
        client = self._async_sdk(api_key)
        start = time.perf_counter()

        try:
            result = await client.audio.translations.create(
                file=audio_file,  # type: ignore[arg-type]
                model=model,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)

        text = result.text if hasattr(result, "text") else str(result)
        return TranscriptionResult(
            text=text,
            model=model,
            latency=time.perf_counter() - start,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
            language=None,
            task="translate",
        )
