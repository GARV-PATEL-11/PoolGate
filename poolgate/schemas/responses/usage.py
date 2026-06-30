from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from poolgate.schemas.common import utcnow


class TokenUsage(BaseModel):
    """Prompt/completion/total token counts for a single provider call."""

    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)
    total_tokens: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _compute_total_if_missing(self) -> "TokenUsage":
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class RequestUsage(BaseModel):
    """Usage record tied to one PoolGateRequest."""

    request_id: str
    model: str
    api_key_id: str | None = None
    token_usage: TokenUsage = Field(default_factory=lambda: TokenUsage())
    call_count: int = Field(1, ge=1)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    latency_ms: float = Field(..., ge=0)
    recorded_at: datetime = Field(default_factory=utcnow)


class QuotaStatus(BaseModel):
    """Point-in-time quota state for a single API key."""

    api_key_id: str
    window_start: datetime
    window_end: datetime

    requests_used: int = Field(0, ge=0)
    requests_limit: int | None = Field(default=None, ge=0)
    remaining_requests: int | None = None

    tokens_used: int = Field(0, ge=0)
    tokens_limit: int | None = Field(default=None, ge=0)
    remaining_tokens: int | None = None

    exhausted: bool = False

    @model_validator(mode="after")
    def _compute_remaining_and_exhausted(self) -> "QuotaStatus":
        if self.requests_limit is not None and self.remaining_requests is None:
            self.remaining_requests = max(self.requests_limit - self.requests_used, 0)
        if self.tokens_limit is not None and self.remaining_tokens is None:
            self.remaining_tokens = max(self.tokens_limit - self.tokens_used, 0)

        if not self.exhausted:
            requests_gone = self.remaining_requests is not None and self.remaining_requests <= 0
            tokens_gone = self.remaining_tokens is not None and self.remaining_tokens <= 0
            self.exhausted = requests_gone or tokens_gone

        return self
