"""
transcription.py — Speech-to-text schemas for Whisper models.

DESIGN NOTE: raw audio bytes are NOT embedded in TranscriptionRequest. This
schema is meant for JSON request bodies — embedding binary audio would mean
base64-inflating it by ~33% and bloating the JSON payload. The intended
pattern is a multipart/form-data upload (audio file + this schema's other
fields as form fields), matching clients/transcription_client.py's
`audio_file: BinaryIO | bytes | tuple[str, bytes]` parameter, which is
populated from the upload out-of-band of this schema.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.common import Metadata, utcnow


class TranscriptionRequest(BaseModel):
	"""
	Request body (non-audio fields) for a transcription or translation call.

	Set `translate=True` to route to the /audio/translations endpoint instead
	of /audio/transcriptions — output is then always English regardless of
	the source language, and `language` is ignored.
	"""

	model_config = ConfigDict(extra="forbid")

	request_type: Literal["transcription"] = "transcription"

	model: str
	audio_filename: str | None = Field(
		default=None, description="Original filename, used for MIME-type inference on upload.",
		)
	audio_format: str | None = Field(default=None, description="e.g. 'mp3', 'wav', 'm4a'.")

	language: str | None = Field(
		default=None,
		description="BCP-47 source language tag, e.g. 'en'. Skips auto-detection. Ignored if translate=True.",
		)
	prompt: str | None = Field(
		default=None, description="Optional vocabulary/spelling priming text.",
		)
	response_format: Literal["text", "json", "verbose_json"] = "text"
	temperature: float = Field(0.0, ge=0.0, le=1.0)
	translate: bool = Field(
		False, description="If True, translate to English instead of transcribing in-language.",
		)
	timeout: float | None = Field(default=None, gt=0)

	metadata: Metadata | None = None


class TranscriptionResponse(BaseModel):
	"""Response body for a transcription or translation call."""

	response_type: Literal["transcription"] = "transcription"

	id: str = Field(..., description="The originating request_id.")
	model: str
	text: str
	language: str | None = Field(
		default=None,
		description="Detected/declared source language. None for translate (output is always English).",
		)
	task: Literal["transcribe", "translate"] = "transcribe"
	duration_seconds: float | None = Field(default=None, ge=0)
	segments: list[dict[str, Any]] | None = Field(
		default=None,
		description="Word/segment-level timestamps, populated only when response_format='verbose_json'.",
		)
	latency_ms: float = Field(..., ge=0)
	created_at: datetime = Field(default_factory=utcnow)


class TranslationRequest(TranscriptionRequest):
	request_type: Literal["transcription"] = "transcription"
	translate: bool = True


class TranslationResponse(TranscriptionResponse):
	task: Literal["translate"] = "translate"
	language: str | None = None
