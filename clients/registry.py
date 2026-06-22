"""
registry.py — Model → capability mapping for PoolGate.

MODEL_CAPABILITIES maps every model ID in MODEL_REGISTRY to the set of
capability strings it supports.  Capability strings correspond 1-to-1 with
the ABC names in capabilities.py:

  Capability string   ABC                          Concrete client
  ─────────────────────────────────────────────────────────────────────────
  "chat"              TextGenerationCapability     ChatClient
  "structured"        StructuredGenerationCap.     StructuredClient
  "tools"             ToolCallingCapability        ToolClient
  "moderation"        ModerationCapability         ModerationClient
  "transcription"     TranscriptionCapability      TranscriptionClient
  "synthesis"         SynthesisCapability          SynthesisClient

──────────────────────────────────────────────────────────────────────────
Capability notes per model family:

  Chat LLMs (allam, llama, gpt-oss, qwen)
    Most support "chat" + "structured" + "tools".
    allam-2-7b is chat-only — no published structured / tool-calling docs yet.

  Compound / Agentic (groq/compound*)
    Support "chat" + "tools" — orchestration-level models.
    Excluded from "structured": these models do not reliably honour
    JSON-schema constraints at time of writing.

  Moderation (prompt-guard, safeguard)
    "moderation" only — passing these to a ChatClient will produce
    garbled output because they are classification fine-tunes, not
    instruction followers.

  Transcription (whisper-*)
    "transcription" only — STT models, audio-in / text-out.
    Accessed via /audio/transcriptions and /audio/translations endpoints.

  Synthesis (orpheus-*)
    "synthesis" only — TTS models, text-in / audio-out.
    Accessed via /audio/speech endpoint.
──────────────────────────────────────────────────────────────────────────

Public API:
  MODEL_CAPABILITIES   dict[str, set[Capability]]
  assert_capability()  raises CapabilityError if capability not supported
  models_for()         returns sorted list of models supporting a capability
"""

from __future__ import annotations

from typing import Literal

from exceptions.request import CapabilityError


# ---------------------------------------------------------------------------
# Type alias — keeps MODEL_CAPABILITIES strongly typed
# ---------------------------------------------------------------------------

Capability = Literal[
	"chat",
	"structured",
	"tools",
	"moderation",
	"transcription",
	"synthesis",
]

# SYNC REQUIREMENT: MODEL_CAPABILITIES must contain the same model IDs as
# llm_models.MODEL_REGISTRY. When adding a new model, update BOTH registries.
# tests/unit/test_llm_models.py::test_model_registry_and_capabilities_have_same_keys
# enforces this contract in CI.
MODEL_CAPABILITIES: dict[str, set[Capability]] = {
	# ── Chat / Instruction LLMs ──────────────────────────────────────────
	"allam-2-7b": {
		"chat",
		# structured / tools withheld — no official support docs at time of writing
		},
	"llama-3.1-8b-instant": {
		"chat",
		"structured",
		"tools",
		},
	"llama-3.3-70b-versatile": {
		"chat",
		"structured",
		"tools",
		},
	"meta-llama/llama-4-scout-17b-16e-instruct": {
		"chat",
		"structured",
		"tools",
		},
	"openai/gpt-oss-20b": {
		"chat",
		"structured",
		"tools",
		},
	"openai/gpt-oss-120b": {
		"chat",
		"structured",
		"tools",
		},
	"qwen/qwen3-32b": {
		"chat",
		"structured",
		"tools",
		},
	"qwen/qwen3.6-27b": {
		"chat",
		"structured",
		"tools",
		},
	# ── Compound / Agentic Models ─────────────────────────────────────────
	"groq/compound": {
		"chat",
		"tools",
		# "structured" excluded — JSON-schema mode unreliable on compound models
		},
	"groq/compound-mini": {
		"chat",
		"tools",
		},
	# ── Moderation / Safety Models ────────────────────────────────────────
	"meta-llama/llama-prompt-guard-2-22m": {
		"moderation",
		},
	"meta-llama/llama-prompt-guard-2-86m": {
		"moderation",
		},
	"openai/gpt-oss-safeguard-20b": {
		"moderation",
		},
	# ── Speech-to-Text (Transcription) ───────────────────────────────────
	"whisper-large-v3": {
		"transcription",
		},
	"whisper-large-v3-turbo": {
		"transcription",
		},
	# ── Text-to-Speech (Synthesis) ────────────────────────────────────────
	"canopylabs/orpheus-arabic-saudi": {
		"synthesis",
		},
	"canopylabs/orpheus-v1-english": {
		"synthesis",
		},
	}


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------


def assert_capability(model: str, capability: Capability) -> None:
	"""
	Raise CapabilityError if model does not support the requested capability.

	Call this at the top of any capability method when you want early,
	descriptive errors instead of cryptic Groq API rejections.

	Example:
		assert_capability("whisper-large-v3", "chat")
		# → CapabilityError: Model 'whisper-large-v3' does not support 'chat'.
		#   Supported: {'transcription'}
	"""
	caps = MODEL_CAPABILITIES.get(model)
	if caps is None:
		raise CapabilityError(
			f"Model '{model}' is not registered in PoolGate. "
			f"Known models: {sorted(MODEL_CAPABILITIES)}",
			model_id=model,
			capability=capability,
			supported_capabilities=[],
			)
	if capability not in caps:
		raise CapabilityError(
			f"Model '{model}' does not support the '{capability}' capability. "
			f"Supported capabilities for this model: {sorted(caps)}",
			model_id=model,
			capability=capability,
			supported_capabilities=sorted(caps),
			)


def models_for(capability: Capability) -> list[str]:
	"""
	Return a sorted list of all model IDs that support a given capability.

	Example:
		models_for("moderation")
		# → ['meta-llama/llama-prompt-guard-2-22m',
		#     'meta-llama/llama-prompt-guard-2-86m',
		#     'openai/gpt-oss-safeguard-20b']
	"""
	return sorted(m for m, caps in MODEL_CAPABILITIES.items() if capability in caps)
