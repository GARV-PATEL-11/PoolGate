"""
llm_models.py
-----------
Shared types for the tracking package. Kept deliberately small — every
tracker depends on these, but never on each other — so each file in
tracking/ stays independently testable and swappable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def today_str(tz: timezone = timezone.utc) -> str:
    """
    Calendar-day key in ISO format, e.g. '2026-06-18'.

    UTC by default so every tracker (and every process, if PoolGate ever
    runs across machines) agrees on when "today" rolls over, regardless of
    server timezone. This is the *calendar* boundary used for reporting —
    it has nothing to do with the rolling windows used for rate limits.
    """
    return datetime.now(tz).date().isoformat()


@dataclass
class TokenUsage:
    """A simple in/out token pair, reusable anywhere a single number isn't enough."""

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
    """
    One calendar day's worth of activity. Plain numbers only — this is
    exactly what gets handed to persistence.py, so keep it JSON-friendly:
    no datetimes, no nested logic, just counters for a single date.
    """

    date: str
    requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "requests": self.requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DailyBucket:
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
    metadata: dict = field(default_factory=dict)


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
