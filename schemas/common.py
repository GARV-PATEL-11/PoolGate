"""
common.py — Foundational types shared across the schemas package.

Everything here is intentionally tiny and dependency-free so every other file
in schemas/ can import from it without risk of circular imports.

NOTE ON DUPLICATION: FinishReason is also defined as a proper Enum class in
the `models` package (models/base.py) and used by clients/ as `FinishReason.STOP`
etc. The Literal alias here is the schema-layer (wire/JSON) representation —
deliberately a plain string Literal rather than an Enum import, so schemas/
has zero dependency on models/ or clients/. If you'd rather have one canonical
definition, models.FinishReason could import from here instead, or vice versa.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """Timezone-aware current UTC timestamp. Used as the default_factory across schemas/."""
    return datetime.now(timezone.utc)


def utc_now() -> datetime:
    """Spec-compatible alias for utcnow()."""
    return utcnow()


UTCTimestamp = datetime


class Region(str, Enum):
    """Deployment/account region hint."""

    GLOBAL = "global"
    US = "us"
    EU = "eu"
    APAC = "apac"


# Intentional duplication: schemas/runtime.py also defines FinishReason.
# runtime.py is SDK-adjacent (used by clients/ and provider_service.py directly).
# This copy is used by the public envelope schemas (envelope.py, structured.py, etc.).
# Both must stay in sync if new values are added.
class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    UNKNOWN = "unknown"


class Metadata(BaseModel):
    """
    Generic, extensible metadata envelope embeddable in any request or response.

    `tags` is for short, indexable string labels (customer name, environment,
    feature flag) you might want to filter or group by later. `extra` is an
    escape hatch for anything else that doesn't deserve a first-class field.

    Embed this as `metadata: Metadata | None = None` on any schema that needs
    free-form annotation without polluting its core fields.
    """

    tags: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
    source: str | None = Field(
        default=None,
        description="Originating component/service that attached this metadata, e.g. 'service.py'.",
    )
    created_at: datetime = Field(default_factory=utcnow)
