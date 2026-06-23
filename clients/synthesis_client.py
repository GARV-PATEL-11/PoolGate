"""
synthesis_client.py — SynthesisClient

Text-to-speech audio synthesis via the Groq audio speech API (Orpheus models).

Supported models:
  canopylabs/orpheus-arabic-saudi   — Arabic (Saudi dialect)
  canopylabs/orpheus-v1-english     — English

synthesize / async_synthesize
  Converts a text string to spoken audio bytes.
  Returns a SynthesisResult containing the raw audio payload, voice metadata,
  and latency.

Public methods:
  synthesize()   — blocking TTS synthesis
  async_synthesize()  — async TTS synthesis
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from clients.base import _new_rid, BaseGroqClient
from clients.capabilities import SynthesisCapability


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class SynthesisResult:
	"""Structured result from a TTS synthesis call."""

	audio: bytes  # raw audio payload — write to file or stream directly
	model: str
	voice: str  # speaker voice used
	response_format: str  # "mp3" | "wav" | "flac" | "opus" | "aac" | "pcm"
	latency: float
	session_id: str
	request_id: str
	api_key_id: str


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SynthesisClient(BaseGroqClient, SynthesisCapability):
	"""
	Stateless client for Orpheus text-to-speech synthesis.

	voice
		Speaker voice identifier — model-specific string.
		Refer to Groq's voice catalogue for each Orpheus model.

	response_format
		Audio container / codec for the returned bytes:
		  "mp3"   — default, widest compatibility
		  "wav"   — uncompressed PCM
		  "flac"  — lossless compressed
		  "opus"  — low-latency streaming codec
		  "aac"   — Apple / mobile-friendly
		  "pcm"   — raw 16-bit little-endian samples

	speed
		Playback speed multiplier.  Accepted range: 0.25 – 4.0.  Default: 1.0.

	Usage example
	-------------
		result = await client.async_synthesize(
			api_key=key,
			model="canopylabs/orpheus-v1-english",
			text="Hello, world.",
			voice="zoe",
			session_id=sid,
			api_key_id=kid,
		)
		Path("output.mp3").write_bytes(result.audio)
	"""

	# ------------------------------------------------------------------
	# Sync
	# ------------------------------------------------------------------

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
		"""
		Blocking TTS synthesis.

		Returns a SynthesisResult.  Write .audio to a file or pipe it to an
		audio player — the bytes are a fully valid audio file in response_format.
		"""
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

	# ------------------------------------------------------------------
	# Async
	# ------------------------------------------------------------------

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
		"""
		Async TTS synthesis.

		Identical contract to synthesize() — uses the native async Groq SDK.
		Designed for use inside FastAPI / asyncio request handlers.
		"""
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
