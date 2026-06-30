from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from poolgate.schemas.common import Metadata, utcnow


class TranscriptionRequest(BaseModel):
    """Request body (non-audio fields) for a transcription or translation call."""

    model_config = ConfigDict(extra="forbid")

    request_type: Literal["transcription"] = "transcription"

    model: str
    audio_filename: str | None = None
    audio_format: str | None = None

    language: str | None = None
    prompt: str | None = None
    response_format: Literal["text", "json", "verbose_json"] = "text"
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    translate: bool = False
    timeout: float | None = Field(default=None, gt=0)

    metadata: Metadata | None = None


class TranscriptionResponse(BaseModel):
    """Response body for a transcription or translation call."""

    response_type: Literal["transcription"] = "transcription"

    id: str = Field(..., description="The originating request_id.")
    model: str
    text: str
    language: str | None = None
    task: Literal["transcribe", "translate"] = "transcribe"
    duration_seconds: float | None = Field(default=None, ge=0)
    segments: list[dict[str, Any]] | None = None
    latency_ms: float = Field(..., ge=0)
    created_at: datetime = Field(default_factory=utcnow)


class TranslationRequest(TranscriptionRequest):
    request_type: Literal["transcription"] = "transcription"
    translate: bool = True


class TranslationResponse(TranscriptionResponse):
    task: Literal["translate"] = "translate"
    language: str | None = None
