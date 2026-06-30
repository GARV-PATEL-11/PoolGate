"""Canonical token usage and quota value objects — no I/O, no Pydantic."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TokenUsage:
    """Canonical token count triple. Consolidates duplicates across schemas and tracking."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class QuotaState:
    """Point-in-time quota utilization snapshot for a single API key."""

    api_key_id: str
    requests_used: int = 0
    requests_limit: int | None = None
    tokens_used: int = 0
    tokens_limit: int | None = None
    exhausted: bool = False
    window_start: datetime = field(default_factory=datetime.utcnow)
    window_end: datetime = field(default_factory=datetime.utcnow)
