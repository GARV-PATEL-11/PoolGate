from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from poolgate.schemas.common import Metadata, utcnow


class APIKey(BaseModel):
    """A single managed Groq API key tracked by PoolGate's key pool."""

    key_id: str
    key_value: SecretStr
    provider: str = "groq"
    status: Literal["active", "disabled", "rate_limited", "revoked"] = "active"

    created_at: datetime = Field(default_factory=utcnow)
    disabled_at: datetime | None = None
    disabled_reason: str | None = None
    rate_limit_reset_at: datetime | None = None
    allowed_models: list[str] | None = None
    metadata: Metadata | None = None


class AccountIdentity(BaseModel):
    account_id: str
    provider: str = "groq"
    alias: str | None = None
    metadata: Metadata | None = None


class APIKeyIdentity(BaseModel):
    key_id: str
    provider: str = "groq"
    account_alias: str | None = None
    account: AccountIdentity | None = None
    metadata: Metadata | None = None
