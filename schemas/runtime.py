"""
Runtime service models used by clients, schedulers, and the public facade.

These are deliberately small, SDK-adjacent models: request parameters,
provider responses, batch summaries, and key status.  The richer files in
schemas/ describe request/response envelopes for service boundaries.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from exceptions.request import InvalidMessageRoleError


class APIKeyStatus(str, Enum):
	ACTIVE = "active"
	RATE_LIMITED = "rate_limited"
	COOLDOWN = "cooldown"
	FAILED = "failed"
	DISABLED = "disabled"
	EXHAUSTED = "exhausted"


# Intentional duplication: schemas/common.py also defines FinishReason.
# This copy is SDK-adjacent and used by clients/ and provider_service.py.
# schemas/common.py's copy is used by the public envelope schemas.
# Both must stay in sync if new values are added.
class FinishReason(str, Enum):
	STOP = "stop"
	LENGTH = "length"
	TOOL_CALLS = "tool_calls"
	CONTENT_FILTER = "content_filter"
	UNKNOWN = "unknown"


class RequestConfig(BaseModel):
	"""Parameters forwarded to Groq completion endpoints."""

	temperature: float = Field(default=1.0, ge=0.0, le=2.0)
	top_p: float = Field(default=1.0, gt=0.0, le=1.0)
	max_tokens: int | None = Field(default=None, ge=1)
	timeout: float = Field(default=30.0, gt=0)
	retries: int = Field(default=3, ge=0, le=10)
	seed: int | None = None
	stream: bool = False
	stop: list[str] | str | None = None

	model_config = {"frozen": False}


# Intentional duplication: schemas/usage.py (public envelope) and tracking/models.py
# (internal dataclass) also define TokenUsage. This SDK-adjacent copy is used by
# clients/ and provider_service.py for immediate response parsing.
class TokenUsage(BaseModel):
	prompt_tokens: int = Field(0, ge=0)
	completion_tokens: int = Field(0, ge=0)
	total_tokens: int = Field(0, ge=0)

	def __add__(self, other: TokenUsage) -> TokenUsage:
		return TokenUsage(
			prompt_tokens=self.prompt_tokens + other.prompt_tokens,
			completion_tokens=self.completion_tokens + other.completion_tokens,
			total_tokens=self.total_tokens + other.total_tokens,
			)


class GroqResponse(BaseModel):
	"""Unified response returned by PoolGate chat-style methods."""

	text: str
	model: str
	usage: TokenUsage
	latency: float
	session_id: str
	request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
	api_key_id: str
	finish_reason: FinishReason = FinishReason.STOP
	metadata: dict[str, Any] = Field(default_factory=dict)
	raw_response: Any | None = None

	model_config = {"arbitrary_types_allowed": True}


class RuntimeChatMessage(BaseModel):
	role: str
	content: str

	@field_validator("role")
	@classmethod
	def validate_role(cls, value: str) -> str:
		allowed = {"system", "user", "assistant", "tool"}
		if value not in allowed:
			raise InvalidMessageRoleError(
				f"role must be one of {sorted(allowed)}, got {value!r}.",
				role=value,
				allowed_roles=allowed,
				)
		return value


class BatchResult(BaseModel):
	index: int
	response: GroqResponse | None = None
	error: str | None = None
	success: bool = True


class BatchSummary(BaseModel):
	total: int
	succeeded: int
	failed: int
	results: list[BatchResult]
	total_tokens: TokenUsage
	total_latency: float
