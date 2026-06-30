"""Foundational types shared across the schemas package."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """Timezone-aware current UTC timestamp."""
    return datetime.now(timezone.utc)


def utc_now() -> datetime:
    return utcnow()


UTCTimestamp = datetime


class Region(str, Enum):
    GLOBAL = "global"
    US = "us"
    EU = "eu"
    APAC = "apac"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


class Metadata(BaseModel):
    """Generic extensible metadata embeddable in any request or response."""

    tags: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
