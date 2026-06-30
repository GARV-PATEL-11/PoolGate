from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from poolgate.schemas.common import Metadata, utcnow
from poolgate.schemas.common.ops import RetryPolicy
from poolgate.schemas.responses.usage import TokenUsage


class RequestType(str, Enum):
    CHAT = "chat"
    STRUCTURED = "structured"
    MODERATION = "moderation"
    TRANSCRIPTION = "transcription"
    SYNTHESIS = "synthesis"


class RequestContext(BaseModel):
    """Identity and tracing metadata for a single request."""

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    api_key_id: str | None = None
    user_id: str | None = None
    trace_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    metadata: Metadata | None = None


class RequestOptions(BaseModel):
    """Caller-tunable behavior controlling HOW a request is routed/handled."""

    retry_policy: RetryPolicy | None = None
    priority: Literal["low", "normal", "high"] = "normal"
    preferred_api_key_ids: list[str] | None = None
    excluded_api_key_ids: list[str] | None = None
    fallback_models: list[str] | None = None
    cache: bool = True
    idempotency_key: str | None = None


class Session(BaseModel):
    """Multi-turn conversation/session state."""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    message_count: int = Field(0, ge=0)
    total_usage: TokenUsage = Field(default_factory=lambda: TokenUsage())
    expires_at: datetime | None = None
    metadata: Metadata | None = None

    def touch(self) -> None:
        self.updated_at = utcnow()
        self.message_count += 1
