"""Unit tests for schemas/synthesis.py — SynthesisRequest/Response validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.synthesis import SynthesisRequest, SynthesisResponse


class TestSynthesisRequest:

	def test_valid_synthesis_request(self):
		req = SynthesisRequest(
			model="playai-tts",
			text="Hello world",
			voice="Arya-PlayAI",
		)
		assert req.model == "playai-tts"
		assert req.response_format == "mp3"

	def test_empty_text_raises(self):
		with pytest.raises(ValidationError):
			SynthesisRequest(model="playai-tts", text="", voice="Arya-PlayAI")

	def test_speed_out_of_range_raises(self):
		with pytest.raises(ValidationError):
			SynthesisRequest(model="playai-tts", text="hi", voice="v", speed=5.0)


class TestSynthesisResponseValidator:

	def test_valid_response_with_audio_url(self):
		resp = SynthesisResponse(
			id="req-1",
			model="playai-tts",
			voice="Arya-PlayAI",
			response_format="mp3",
			audio_url="https://storage.example.com/audio/req-1.mp3",
			latency_ms=300.0,
		)
		assert resp.audio_url is not None
		assert resp.audio_base64 is None

	def test_valid_response_with_audio_base64(self):
		resp = SynthesisResponse(
			id="req-1",
			model="playai-tts",
			voice="Arya-PlayAI",
			response_format="mp3",
			audio_base64="dGVzdA==",
			latency_ms=300.0,
		)
		assert resp.audio_base64 is not None
		assert resp.audio_url is None

	def test_both_audio_fields_set_raises(self):
		with pytest.raises(ValidationError) as exc_info:
			SynthesisResponse(
				id="req-1",
				model="playai-tts",
				voice="Arya-PlayAI",
				response_format="mp3",
				audio_url="https://example.com/a.mp3",
				audio_base64="dGVzdA==",
				latency_ms=300.0,
			)
		assert "audio" in str(exc_info.value).lower()

	def test_neither_audio_field_set_raises(self):
		with pytest.raises(ValidationError) as exc_info:
			SynthesisResponse(
				id="req-1",
				model="playai-tts",
				voice="Arya-PlayAI",
				response_format="mp3",
				latency_ms=300.0,
			)
		assert "audio" in str(exc_info.value).lower()
