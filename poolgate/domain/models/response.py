"""Pure domain value objects for provider responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from poolgate.domain.models.usage import TokenUsage


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


@dataclass
class ProviderResponse:
    """Generic response envelope returned by any provider adapter."""

    text: str
    model: str
    finish_reason: FinishReason = FinishReason.STOP
    latency: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResult:
    """Structured completion with full context for upstream consumers."""

    response: ProviderResponse
    request_id: str
    session_id: str
    api_key_id: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
