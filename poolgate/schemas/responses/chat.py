from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from poolgate.schemas.common import FinishReason, utcnow
from poolgate.schemas.requests.chat import ChatMessage
from poolgate.schemas.responses.usage import TokenUsage


class ChatResponse(BaseModel):
    """Response body for a chat completion."""

    response_type: Literal["chat"] = "chat"

    id: str = Field(..., description="The originating request_id.")
    model: str
    message: ChatMessage
    finish_reason: FinishReason
    usage: TokenUsage
    latency_ms: float = Field(..., ge=0)
    created_at: datetime = Field(default_factory=utcnow)
