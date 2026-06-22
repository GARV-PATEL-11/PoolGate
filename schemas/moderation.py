"""
moderation.py — Safety classification schemas for prompt-guard / safeguard models.

See clients/moderation_client.py for the underlying label conventions:
  Prompt Guard:        SAFE | JAILBREAK | INDIRECT
  GPT-OSS-Safeguard:   safe | unsafe
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.chat import ChatMessage
from schemas.common import Metadata, utcnow
from schemas.usage import TokenUsage


# Labels (case-insensitive) treated as "flagged" when computing ModerationResponse.flagged.
_UNSAFE_LABELS = frozenset({"unsafe", "jailbreak", "indirect"})


class ModerationRequest(BaseModel):
	"""Request body for a content safety classification call."""

	model_config = ConfigDict(extra="forbid")

	request_type: Literal["moderation"] = "moderation"

	model: str
	text: str = Field(..., min_length=1)
	context: list[ChatMessage] | None = Field(
		default=None,
		description="Optional preceding conversation turns, for moderation flows that consider prior context.",
		)
	timeout: float | None = Field(default=None, gt=0)

	metadata: Metadata | None = None


class ModerationResponse(BaseModel):
	"""
	Response body for a content safety classification call.

	`flagged` is a derived convenience boolean — True when `label` (case-
	insensitively) indicates unsafe content, so callers don't need to know
	each model's specific label vocabulary to branch on the verdict.
	"""

	response_type: Literal["moderation"] = "moderation"

	id: str = Field(..., description="The originating request_id.")
	model: str
	label: str
	flagged: bool = Field(
		default=False,
		description="Auto-computed from `label` if left at its default — set explicitly to override.",
		)
	raw_text: str = Field(
		..., description="Full model output — may include a brief rationale after the label.",
		)
	usage: TokenUsage
	latency_ms: float = Field(..., ge=0)
	created_at: datetime = Field(default_factory=utcnow)

	@model_validator(mode="after")
	def _compute_flagged(self) -> ModerationResponse:
		if not self.flagged:
			self.flagged = self.label.strip().lower() in _UNSAFE_LABELS
		return self
