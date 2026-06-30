from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from poolgate.schemas.common import FinishReason, Metadata, utcnow
from poolgate.schemas.requests.chat import ChatMessage
from poolgate.schemas.responses.usage import TokenUsage


class StructuredRequest(BaseModel):
    """Request body for schema-constrained or plain-JSON-mode generation."""

    model_config = ConfigDict(extra="forbid")

    request_type: Literal["structured"] = "structured"

    model: str
    messages: list[ChatMessage] = Field(..., min_length=1)
    json_schema: dict[str, Any] | None = None
    schema_name: str | None = None
    strict: bool = True

    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = None
    timeout: float | None = Field(default=None, gt=0)

    metadata: Metadata | None = None

    @model_validator(mode="after")
    def _default_schema_name(self) -> "StructuredRequest":
        if self.json_schema is not None and self.schema_name is None:
            self.schema_name = self.json_schema.get("title", "response")
        return self


class StructuredResponse(BaseModel):
    """Response body for a structured generation call."""

    response_type: Literal["structured"] = "structured"

    id: str = Field(..., description="The originating request_id.")
    model: str
    data: dict[str, Any]
    raw_text: str
    finish_reason: FinishReason
    usage: TokenUsage
    latency_ms: float = Field(..., ge=0)
    created_at: datetime = Field(default_factory=utcnow)
