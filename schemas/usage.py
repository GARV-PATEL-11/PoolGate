"""
usage.py — Token accounting and quota tracking schemas.

These back the tracking/ package (token_tracker.py, quota_tracker.py,
account_tracker.py, rolling_window.py) at the schema layer.

NOTE ON DUPLICATION: TokenUsage already exists in the `models` package and is
embedded directly in GroqResponse (see clients/base.py:_parse_usage). That
version is the raw per-SDK-call shape returned straight off the wire. This
TokenUsage is the schema-layer twin — same fields, used wherever usage needs
to be validated/serialized at the service boundary (e.g. inside RequestUsage,
Session, PoolGateResponse). If you want a single source of truth, models.TokenUsage
could import this one instead of redefining it.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from schemas.common import utcnow


# Intentional duplication: schemas/runtime.py and tracking/models.py also define TokenUsage.
# This version (schemas/usage.py) is the public envelope schema with auto-fill validation.
# schemas/runtime.py's copy is SDK-adjacent (used by clients/ layer).
# tracking/models.py's copy is a plain dataclass used internally by trackers.
class TokenUsage(BaseModel):
    """Prompt/completion/total token counts for a single provider call."""

    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)
    total_tokens: int = Field(0, ge=0)

    @model_validator(mode="after")
    def _compute_total_if_missing(self) -> TokenUsage:
        """Auto-fill total_tokens when the caller only supplied the two halves."""
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Allow `usage_a + usage_b` to accumulate totals, e.g. across a tool-calling loop."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class RequestUsage(BaseModel):
    """
    Usage record tied to one PoolGateRequest, which may span multiple
    underlying provider calls (e.g. a multi-turn tool-calling loop, or a
    request retried after a transient error).

    `call_count` is the number of provider round-trips this single logical
    request triggered — 1 for a plain chat call, 2+ for tool loops or retries.
    """

    request_id: str
    model: str
    api_key_id: str | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    call_count: int = Field(1, ge=1)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    latency_ms: float = Field(..., ge=0)
    recorded_at: datetime = Field(default_factory=utcnow)


class QuotaStatus(BaseModel):
    """
    Point-in-time quota state for a single API key over a rolling/fixed window.

    Supply `requests_used`/`tokens_used` plus the optional limits; remaining_*
    and `exhausted` are computed automatically if not provided directly.
    """

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
    def _compute_remaining_and_exhausted(self) -> QuotaStatus:
        if self.requests_limit is not None and self.remaining_requests is None:
            self.remaining_requests = max(self.requests_limit - self.requests_used, 0)
        if self.tokens_limit is not None and self.remaining_tokens is None:
            self.remaining_tokens = max(self.tokens_limit - self.tokens_used, 0)

        if not self.exhausted:
            requests_gone = self.remaining_requests is not None and self.remaining_requests <= 0
            tokens_gone = self.remaining_tokens is not None and self.remaining_tokens <= 0
            self.exhausted = requests_gone or tokens_gone

        return self
