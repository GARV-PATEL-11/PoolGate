"""SynthesisCapability — text-to-speech audio synthesis."""

from __future__ import annotations

import time
from dataclasses import dataclass

from poolgate.providers.base import _new_rid
from poolgate.providers.groq.capabilities import SynthesisCapability as _SynthesisABC
from poolgate.providers.groq.client import GroqProvider


@dataclass
class SynthesisResult:
    """Structured result from a TTS synthesis call."""

    audio: bytes
    model: str
    voice: str
    response_format: str
    latency: float
    session_id: str
    request_id: str
    api_key_id: str


class SynthesisCapability(GroqProvider, _SynthesisABC):
    """Stateless client for Orpheus text-to-speech synthesis."""

    def synthesize(
        self,
        api_key: str,
        model: str,
        text: str,
        voice: str,
        session_id: str,
        api_key_id: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        timeout: float | None = None,
        request_id: str | None = None,
    ) -> SynthesisResult:
        rid = _new_rid(request_id)
        client = self._sync_sdk(api_key)
        start = time.perf_counter()
        try:
            response = client.audio.speech.create(
                model=model,
                voice=voice,  # type: ignore[arg-type]
                input=text,
                response_format=response_format,  # type: ignore[arg-type]
                speed=speed,
                timeout=timeout,
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise
        return SynthesisResult(
            audio=response.read(),
            model=model,
            voice=voice,
            response_format=response_format,
            latency=time.perf_counter() - start,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
        )

    async def async_synthesize(
        self,
        api_key: str,
        model: str,
        text: str,
        voice: str,
        session_id: str,
        api_key_id: str,
        response_format: str = "mp3",
        speed: float = 1.0,
        timeout: float | None = None,
        request_id: str | None = None,
    ) -> SynthesisResult:
        rid = _new_rid(request_id)
        client = self._async_sdk(api_key)
        start = time.perf_counter()
        try:
            response = await client.audio.speech.create(
                model=model,
                voice=voice,  # type: ignore[arg-type]
                input=text,
                response_format=response_format,  # type: ignore[arg-type]
                speed=speed,
                timeout=timeout,
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise
        return SynthesisResult(
            audio=response.read(),
            model=model,
            voice=voice,
            response_format=response_format,
            latency=time.perf_counter() - start,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
        )
