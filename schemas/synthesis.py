"""
synthesis.py — Text-to-speech schemas for Orpheus models.

DESIGN NOTE: SynthesisResponse carries `audio_url` or `audio_base64`, never
raw bytes — unlike clients/synthesis_client.py's SynthesisResult, which
returns `audio: bytes` directly because it's an internal, in-process object.
This schema is for the service boundary, where raw bytes either get persisted
to storage (→ audio_url) or base64-encoded for inline JSON transport
(→ audio_base64). Exactly one should be set.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.common import Metadata, utcnow


class SynthesisRequest(BaseModel):
    """Request body for a text-to-speech synthesis call."""

    model_config = ConfigDict(extra="forbid")

    request_type: Literal["synthesis"] = "synthesis"

    model: str
    text: str = Field(..., min_length=1, max_length=10_000)
    voice: str
    response_format: Literal["mp3", "wav", "flac", "opus", "aac", "pcm"] = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    timeout: float | None = Field(default=None, gt=0)

    metadata: Metadata | None = None


class SynthesisResponse(BaseModel):
    """
    Response body for a text-to-speech synthesis call.

    Exactly one of audio_url / audio_base64 must be set — see module docstring.
    """

    response_type: Literal["synthesis"] = "synthesis"

    id: str = Field(..., description="The originating request_id.")
    model: str
    voice: str
    response_format: str
    audio_url: str | None = Field(
        default=None,
        description="Set when the audio was persisted to storage.",
    )
    audio_base64: str | None = Field(
        default=None,
        description="Set when the audio is returned inline.",
    )
    duration_seconds: float | None = Field(default=None, ge=0)
    latency_ms: float = Field(..., ge=0)
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _validate_exactly_one_audio_field(self) -> SynthesisResponse:
        has_url = self.audio_url is not None
        has_b64 = self.audio_base64 is not None
        if has_url == has_b64:
            raise ValueError("Exactly one of audio_url or audio_base64 must be set.")
        return self
