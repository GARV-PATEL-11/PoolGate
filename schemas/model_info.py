"""
model_info.py — Model registry schemas: capability descriptors and metadata.

ModelCapabilities is the rich, per-model feature descriptor (booleans + limits).
CapabilitySet is a thin wrapper around just the flat capability names, mirroring
clients.registry.MODEL_CAPABILITIES at the schema layer so capability membership
can be serialized over the wire (e.g. a GET /models/{id}/capabilities endpoint).
ModelInfo is the top-level descriptor combining both with registry metadata.

NOTE ON DUPLICATION: the Capability Literal below duplicates
clients.registry.Capability. It's redefined here rather than imported so that
schemas/ has no dependency on clients/ (schemas/ is meant to sit underneath
clients/ in the dependency graph, not beside or above it). If you want one
source of truth, clients/registry.py could import Capability from here instead.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# Schema-layer mirror of clients.registry.Capability (see module docstring above).
Capability = Literal[
	"chat",
	"structured",
	"tools",
	"moderation",
	"transcription",
	"synthesis",
]

_ALL_CAPABILITIES: tuple[Capability, ...] = (
	"chat",
	"structured",
	"tools",
	"moderation",
	"transcription",
	"synthesis",
	)


class CapabilitySet(BaseModel):
	"""
	Flat set of capability names a model supports — the wire-serializable
	twin of a single entry in clients.registry.MODEL_CAPABILITIES.
	"""

	model_id: str
	capabilities: set[Capability]

	def supports(self, capability: Capability) -> bool:
		"""True if this model supports the given capability."""
		return capability in self.capabilities


class ModelCapabilities(BaseModel):
	"""
	Detailed, per-capability feature flags and limits for a single model.

	The six boolean flags below correspond 1-to-1 with the Capability literal
	values and with clients/registry.py's MODEL_CAPABILITIES sets. The fields
	beneath them are capability-specific limits that a flat set can't express —
	e.g. two models can both support "tools" while differing in max_tools.
	"""

	chat: bool = False
	structured: bool = False
	tools: bool = False
	moderation: bool = False
	transcription: bool = False
	synthesis: bool = False

	supports_streaming: bool = Field(
		False, description="True for chat models exposing stream()/async_stream().",
		)
	supports_json_schema: bool = Field(
		False,
		description="True if structured output can be schema-constrained, not just JSON-mode.",
		)
	max_tools: int | None = Field(
		default=None, ge=0, description="Max tool definitions per request, if capped.",
		)
	max_context_tokens: int | None = Field(default=None, ge=0)
	max_output_tokens: int | None = Field(default=None, ge=0)
	supported_languages: list[str] | None = Field(
		default=None,
		description="BCP-47 language codes. Relevant for transcription (source) and synthesis (voice locale).",
		)
	available_voices: list[str] | None = Field(
		default=None, description="Synthesis-only: valid `voice` values for this model.",
		)

	def as_capability_set(self, model_id: str) -> CapabilitySet:
		"""Collapse the boolean flags into a flat CapabilitySet for a given model_id."""
		active = {cap for cap in _ALL_CAPABILITIES if getattr(self, cap)}
		return CapabilitySet(model_id=model_id, capabilities=active)


class ModelInfo(BaseModel):
	"""Top-level registry descriptor for a single model PoolGate can route to."""

	model_id: str
	provider: str = "groq"
	display_name: str
	family: Literal["chat", "compound", "moderation", "transcription", "synthesis"]
	capabilities: ModelCapabilities
	context_window: int | None = Field(default=None, ge=0)
	max_output_tokens: int | None = Field(default=None, ge=0)
	deprecated: bool = False
	description: str | None = None
