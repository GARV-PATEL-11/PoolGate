from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Canonical Capability definition — poolgate/providers/registry.py imports from here.
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
    model_id: str
    capabilities: set[Capability]

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities


class ModelCapabilities(BaseModel):
    chat: bool = False
    structured: bool = False
    tools: bool = False
    moderation: bool = False
    transcription: bool = False
    synthesis: bool = False

    supports_streaming: bool = False
    supports_json_schema: bool = False
    max_tools: int | None = Field(default=None, ge=0)
    max_context_tokens: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, ge=0)
    supported_languages: list[str] | None = None
    available_voices: list[str] | None = None

    def as_capability_set(self, model_id: str) -> CapabilitySet:
        active = {cap for cap in _ALL_CAPABILITIES if getattr(self, cap)}
        return CapabilitySet(model_id=model_id, capabilities=active)


class ModelInfo(BaseModel):
    model_id: str
    provider: str = "groq"
    display_name: str
    family: Literal["chat", "compound", "moderation", "transcription", "synthesis"]
    capabilities: ModelCapabilities
    context_window: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, ge=0)
    deprecated: bool = False
    description: str | None = None
