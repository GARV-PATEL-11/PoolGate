"""
envelope.py — Top-level request/response envelope wrapping any capability payload.

PoolGateRequest.payload and PoolGateResponse.payload are discriminated unions,
resolved at runtime via the `request_type` / `response_type` Literal fields
fixed on each capability schema (ChatRequest, StructuredRequest, ... in their
respective files).

IMPORTANT CAVEAT on discriminated unions: when constructing these directly in
Python (e.g. `PoolGateRequest(payload=ChatRequest(...), ...)`), the discriminator
field's default value resolves the union automatically — you don't need to set
request_type yourself. But when parsing raw JSON/dict input (e.g.
`PoolGateRequest.model_validate(raw_dict)`), Pydantic resolves the union by
reading the discriminator key from the input BEFORE defaults are applied — so
the incoming `payload` dict MUST include an explicit "request_type"/"response_type"
key, even though it's optional when constructing from Python objects directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from schemas.chat import ChatRequest, ChatResponse
from schemas.common import utcnow
from schemas.context import RequestContext, RequestOptions
from schemas.moderation import ModerationRequest, ModerationResponse
from schemas.ops import ErrorResponse
from schemas.structured import StructuredRequest, StructuredResponse
from schemas.synthesis import SynthesisRequest, SynthesisResponse
from schemas.transcription import TranscriptionRequest, TranscriptionResponse
from schemas.usage import RequestUsage

CapabilityRequest = Annotated[
    ChatRequest | StructuredRequest | ModerationRequest | TranscriptionRequest | SynthesisRequest,
    Field(discriminator="request_type"),
]

CapabilityResponse = Annotated[
    ChatResponse | StructuredResponse | ModerationResponse | TranscriptionResponse | SynthesisResponse,
    Field(discriminator="response_type"),
]


class PoolGateRequest(BaseModel):
    """
    Top-level request envelope. `payload` is the capability-specific request
    body; `context` and `options` carry everything about HOW it's handled.
    """

    context: RequestContext = Field(default_factory=RequestContext)
    options: RequestOptions = Field(default_factory=RequestOptions)
    payload: CapabilityRequest

    @property
    def capability(self) -> str:
        """Convenience accessor — equivalent to payload.request_type."""
        return self.payload.request_type


class PoolGateResponse(BaseModel):
    """
    Top-level response envelope. Exactly one of `payload` / `error` is set,
    governed by `success`.
    """

    context: RequestContext
    success: bool = True
    payload: CapabilityResponse | None = None
    error: ErrorResponse | None = None
    usage: RequestUsage | None = None
    total_latency_ms: float = Field(..., ge=0)
    completed_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _validate_success_consistency(self) -> PoolGateResponse:
        if self.success and self.error is not None:
            raise ValueError("success=True responses must not include an error.")
        if not self.success and self.payload is not None:
            raise ValueError("success=False responses must not include a payload.")
        if not self.success and self.error is None:
            raise ValueError("success=False responses must include an error.")
        return self


PoolGetRequest = PoolGateRequest
PoolGetResponse = PoolGateResponse
