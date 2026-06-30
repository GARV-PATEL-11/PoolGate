"""Pure domain value objects for API key state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class KeyStatus(str, Enum):
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    COOLDOWN = "cooldown"
    FAILED = "failed"
    DISABLED = "disabled"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class APIKey:
    """Lightweight value object representing a pooled API key."""

    key_id: str
    masked_key: str

    def __repr__(self) -> str:
        return f"APIKey(key_id={self.key_id!r}, masked={self.masked_key!r})"
