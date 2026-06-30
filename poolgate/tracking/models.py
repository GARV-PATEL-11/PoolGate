"""Shared types for the tracking package."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def today_str(tz: timezone = timezone.utc) -> str:
    return datetime.now(tz).date().isoformat()


@dataclass
class TokenUsage:
    """In/out token pair for tracking use (separate from domain/schemas TokenUsage)."""

    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def total(self) -> int:
        return self.tokens_in + self.tokens_out

    def add(self, tokens_in: int = 0, tokens_out: int = 0) -> None:
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out


@dataclass
class DailyBucket:
    date: str
    requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "requests": self.requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DailyBucket:
        return cls(
            date=d["date"],
            requests=d.get("requests", 0),
            successful_requests=d.get("successful_requests", 0),
            failed_requests=d.get("failed_requests", 0),
            tokens_in=d.get("tokens_in", 0),
            tokens_out=d.get("tokens_out", 0),
        )


@dataclass
class AccountStats:
    account_id: str
    requests: int = 0
    tokens: int = 0
    last_used: str | None = None


@dataclass
class RequestRecord:
    request_id: str
    model: str | None = None
    api_key_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenStats:
    scope: str
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


@dataclass
class QuotaSnapshot:
    scope: str
    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    reset_seconds: float | None = None


@dataclass
class RollingWindowEntry:
    timestamp: float
    weight: int = 1
