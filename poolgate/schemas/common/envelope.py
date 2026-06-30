from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from poolgate.schemas.common import utcnow
from poolgate.schemas.common.context import RequestContext, RequestOptions
from poolgate.schemas.common.ops import ErrorResponse
from poolgate.schemas.requests.chat import ChatRequest
from poolgate.schemas.requests.moderation import ModerationRequest, ModerationResponse
from poolgate.schemas.requests.structured import StructuredRequest, StructuredResponse
from poolgate.schemas.requests.synthesis import SynthesisRequest, SynthesisResponse
from poolgate.schemas.requests.transcription import TranscriptionRequest, TranscriptionResponse
from poolgate.schemas.responses.chat import ChatResponse
from poolgate.schemas.responses.usage import RequestUsage

CapabilityRequest = Annotated[
    ChatRequest | StructuredRequest | ModerationRequest | TranscriptionRequest | SynthesisRequest,
    Field(discriminator="request_type"),
]

CapabilityResponse = Annotated[
    ChatResponse | StructuredResponse | ModerationResponse | TranscriptionResponse | SynthesisResponse,
    Field(discriminator="response_type"),
]


class PoolGateRequest(BaseModel):
    """Top-level request envelope."""

    context: RequestContext = Field(default_factory=RequestContext)
    options: RequestOptions = Field(default_factory=RequestOptions)
    payload: CapabilityRequest

    @property
    def capability(self) -> str:
        return self.payload.request_type


class PoolGateResponse(BaseModel):
    """Top-level response envelope. Exactly one of payload/error is set."""

    context: RequestContext
    success: bool = True
    payload: CapabilityResponse | None = None
    error: ErrorResponse | None = None
    usage: RequestUsage | None = None
    total_latency_ms: float = Field(..., ge=0)
    completed_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _validate_success_consistency(self) -> "PoolGateResponse":
        if self.success and self.error is not None:
            raise ValueError("success=True responses must not include an error.")
        if not self.success and self.payload is not None:
            raise ValueError("success=False responses must not include a payload.")
        if not self.success and self.error is None:
            raise ValueError("success=False responses must include an error.")
        return self


PoolGetRequest = PoolGateRequest
PoolGetResponse = PoolGateResponse
