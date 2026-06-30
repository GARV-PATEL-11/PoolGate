"""TranscriptionCapability — speech-to-text transcription and translation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import BinaryIO

from poolgate.providers.base import _new_rid
from poolgate.providers.groq.capabilities import TranscriptionCapability as _TranscriptionABC
from poolgate.providers.groq.client import GroqProvider


@dataclass
class TranscriptionResult:
    """Structured result from a Whisper transcription or translation call."""

    text: str
    model: str
    latency: float
    session_id: str
    request_id: str
    api_key_id: str
    language: str | None = None
    task: str = "transcribe"


class TranscriptionCapability(GroqProvider, _TranscriptionABC):
    """Stateless client for Whisper-based audio transcription and translation."""

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
            raise
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
            raise
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
            raise
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
            raise
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
