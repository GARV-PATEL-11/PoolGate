"""
context.py — Request-lifecycle context, caller-tunable options, and session state.

RequestContext carries identity/tracing fields threaded through a request's
full lifecycle (service.py → schedulers/ → clients/ → tracking/).
RequestOptions carries behavioral knobs that influence HOW PoolGate handles
a request (retries, key selection, fallback models) — distinct from the
generation parameters that live on each capability's *Request schema.
Session backs session_manager.py's multi-turn conversation tracking.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from schemas.common import Metadata, utcnow
from schemas.ops import RetryPolicy
from schemas.usage import TokenUsage


class RequestType(str, Enum):
	CHAT = "chat"
	STRUCTURED = "structured"
	MODERATION = "moderation"
	TRANSCRIPTION = "transcription"
	SYNTHESIS = "synthesis"


class RequestContext(BaseModel):
	"""
	Identity and tracing metadata for a single request, independent of which
	capability is being invoked. Echoed back unchanged in PoolGateResponse.
	"""

	request_id: str = Field(default_factory=lambda: str(uuid4()))
	session_id: str | None = None
	api_key_id: str | None = None
	user_id: str | None = Field(
		default=None,
		description="End-customer identifier, for multi-tenant billing/quota attribution.",
		)
	trace_id: str | None = Field(
		default=None, description="Distributed tracing correlation ID, e.g. for Langfuse.",
		)
	created_at: datetime = Field(default_factory=utcnow)
	metadata: Metadata | None = None


class RequestOptions(BaseModel):
	"""
	Caller-tunable behavior that controls HOW a request is routed/handled —
	as opposed to generation parameters (temperature, max_tokens, ...), which
	live on the capability-specific *Request schemas in chat.py / structured.py / etc.
	"""

	retry_policy: RetryPolicy | None = Field(
		default=None,
		description="Overrides the service-wide default from retry.py for this request only.",
		)
	priority: Literal["low", "normal", "high"] = "normal"
	preferred_api_key_ids: list[str] | None = Field(
		default=None, description="Pin this request to specific keys, if possible.",
		)
	excluded_api_key_ids: list[str] | None = None
	fallback_models: list[str] | None = Field(
		default=None,
		description="Tried in order if the primary model is unavailable or rate-limited.",
		)
	cache: bool = Field(
		True, description="Whether PoolGate may serve/store a cached response for this request.",
		)
	idempotency_key: str | None = Field(
		default=None,
		description="Caller-supplied key to dedupe retried requests at the service boundary.",
		)


class Session(BaseModel):
	"""
	Multi-turn conversation/session state tracked by session_manager.py.

	Call `.touch()` after appending a new turn to bump message_count and
	updated_at in one step.
	"""

	session_id: str = Field(default_factory=lambda: str(uuid4()))
	user_id: str | None = None
	created_at: datetime = Field(default_factory=utcnow)
	updated_at: datetime = Field(default_factory=utcnow)
	message_count: int = Field(0, ge=0)
	total_usage: TokenUsage = Field(default_factory=TokenUsage)
	expires_at: datetime | None = None
	metadata: Metadata | None = None

	def touch(self) -> None:
		"""Bump updated_at and message_count after a new turn is appended to the session."""
		self.updated_at = utcnow()
		self.message_count += 1
