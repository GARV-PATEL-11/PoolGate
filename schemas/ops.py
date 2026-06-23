"""
ops.py — Operational schemas: health checks, retry policy, error responses.

HealthStatus backs a /health endpoint in main.py. RetryPolicy is consumed by
retry.py and by RequestOptions (context.py) for per-request overrides.
ErrorResponse is the standard error body service.py returns for any failure,
mirroring the structured exceptions in exceptions.py (APIKeyDisabledError,
RateLimitExceededError, etc.) at the HTTP/service boundary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from schemas.common import utcnow


class HealthStatus(BaseModel):
    """Overall PoolGate service health snapshot."""

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str | None = None
    uptime_seconds: float = Field(..., ge=0)
    active_keys: int = Field(..., ge=0)
    disabled_keys: int = Field(..., ge=0)
    checked_at: datetime = Field(default_factory=utcnow)
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional free-form diagnostics, e.g. per-model availability or queue depth.",
    )


class RetryPolicy(BaseModel):
    """
    Exponential backoff retry policy.

    Used as the service-wide default in retry.py, and overridable per request
    via RequestOptions.retry_policy.
    """

    max_attempts: int = Field(3, ge=1, le=10)
    initial_backoff_seconds: float = Field(1.0, gt=0)
    backoff_multiplier: float = Field(2.0, ge=1.0)
    max_backoff_seconds: float = Field(60.0, gt=0)
    jitter: bool = Field(
        True,
        description="Add random jitter to each backoff interval to avoid thundering-herd retries.",
    )
    retry_on: list[Literal["rate_limit", "timeout", "server_error", "auth_error"]] = Field(
        default_factory=lambda: ["rate_limit", "timeout", "server_error"],
        description="auth_error is excluded by default — a disabled key won't recover by retrying.",
    )

    @model_validator(mode="after")
    def _validate_backoff_bounds(self) -> RetryPolicy:
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= initial_backoff_seconds.")
        return self


class ErrorResponse(BaseModel):
    """
    Standard error body returned by the service layer for any failed request.

    error_code is intentionally a free string rather than a Literal — the
    canonical values today mirror exceptions.py: 'rate_limit_exceeded',
    'api_key_disabled', 'invalid_response', plus service-level additions like
    'capability_not_supported', 'validation_error', 'timeout', 'internal_error'.
    Keeping it a str avoids this schema going stale every time exceptions.py grows.
    """

    error_code: str
    message: str
    request_id: str | None = None
    retry_after: float | None = Field(
        default=None,
        ge=0,
        description="Seconds to wait before retrying, populated for rate_limit_exceeded errors.",
    )
    details: dict[str, Any] | None = None
    occurred_at: datetime = Field(default_factory=utcnow)
