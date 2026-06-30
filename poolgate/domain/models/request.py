"""Pure domain value objects for provider requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderRequest:
    """Generic provider request envelope. Provider-agnostic."""

    model: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    temperature: float = 1.0
    max_tokens: int | None = None
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestContext:
    """Request-scoped metadata threaded through the entire call lifecycle."""

    request_id: str
    session_id: str
    model: str
    capability: str = ""
    api_key_id: str = ""
    retry_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
