"""
keys.py — Managed API key schema for the key pool (key_manager/key_pool.py).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from schemas.common import Metadata, utcnow


class APIKey(BaseModel):
	"""
	A single managed Groq API key tracked by PoolGate's key pool.

	`key_value` uses Pydantic's SecretStr rather than a plain str — it's
	automatically masked in repr(), str(), and logging output. Call
	`.key_value.get_secret_value()` when you actually need the raw key
	(e.g. inside clients/base.py:_sync_sdk / _async_sdk).
	"""

	key_id: str
	key_value: SecretStr
	provider: str = "groq"
	status: Literal["active", "disabled", "rate_limited", "revoked"] = "active"

	created_at: datetime = Field(default_factory=utcnow)
	disabled_at: datetime | None = None
	disabled_reason: str | None = None
	rate_limit_reset_at: datetime | None = Field(
		default=None,
		description="If status='rate_limited', when the scheduler should next consider this key.",
		)

	allowed_models: list[str] | None = Field(
		default=None,
		description="Restrict this key to a subset of MODEL_REGISTRY entries. None = all models permitted.",
		)
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
