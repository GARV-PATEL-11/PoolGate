from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from poolgate.schemas.common import Metadata, utcnow
from poolgate.schemas.requests.chat import ChatMessage
from poolgate.schemas.responses.usage import TokenUsage

_UNSAFE_LABELS = frozenset({"unsafe", "jailbreak", "indirect"})


class ModerationRequest(BaseModel):
    """Request body for a content safety classification call."""

    model_config = ConfigDict(extra="forbid")

    request_type: Literal["moderation"] = "moderation"

    model: str
    text: str = Field(..., min_length=1)
    context: list[ChatMessage] | None = None
    timeout: float | None = Field(default=None, gt=0)

    metadata: Metadata | None = None


class ModerationResponse(BaseModel):
    """Response body for a content safety classification call."""

    response_type: Literal["moderation"] = "moderation"

    id: str = Field(..., description="The originating request_id.")
    model: str
    label: str
    flagged: bool = False
    raw_text: str
    usage: TokenUsage
    latency_ms: float = Field(..., ge=0)
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _compute_flagged(self) -> "ModerationResponse":
        if not self.flagged:
            self.flagged = self.label.strip().lower() in _UNSAFE_LABELS
        return self
