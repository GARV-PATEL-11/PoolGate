from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, Field, model_validator

from poolgate.schemas.common import utcnow


class HealthStatus(BaseModel):
    """Overall PoolGate service health snapshot."""

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str | None = None
    uptime_seconds: float = Field(..., ge=0)
    active_keys: int = Field(..., ge=0)
    disabled_keys: int = Field(..., ge=0)
    checked_at: datetime = Field(default_factory=utcnow)
    details: dict[str, Any] | None = None


class RetryPolicy(BaseModel):
    """Exponential backoff retry policy."""

    max_attempts: int = Field(3, ge=1, le=10)
    initial_backoff_seconds: float = Field(1.0, gt=0)
    backoff_multiplier: float = Field(2.0, ge=1.0)
    max_backoff_seconds: float = Field(60.0, gt=0)
    jitter: bool = True
    retry_on: list[Literal["rate_limit", "timeout", "server_error", "auth_error"]] = Field(
        default_factory=lambda: cast(
            list[Literal["rate_limit", "timeout", "server_error", "auth_error"]],
            ["rate_limit", "timeout", "server_error"],
        )
    )

    @model_validator(mode="after")
    def _validate_backoff_bounds(self) -> "RetryPolicy":
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= initial_backoff_seconds.")
        return self


class ErrorResponse(BaseModel):
    """Standard error body returned by the service layer for any failed request."""

    error_code: str
    message: str
    request_id: str | None = None
    retry_after: float | None = Field(default=None, ge=0)
    details: dict[str, Any] | None = None
    occurred_at: datetime = Field(default_factory=utcnow)
