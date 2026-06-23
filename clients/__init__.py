"""
clients/__init__.py — Public surface of the PoolGate clients package.

Import everything you need from here.  Internal modules (base, etc.) are
considered private and should not be imported directly by application code.

── Concrete clients ─────────────────────────────────────────────────────────
  ChatClient           invoke / async_invoke / stream / async_stream
  StructuredClient     invoke_structured / async_invoke_structured
  ToolClient           invoke_tools / async_invoke_tools
  ModerationClient     moderate / async_moderate
  TranscriptionClient  transcribe / async_transcribe / translate / async_translate
  SynthesisClient      synthesize / async_synthesize

── Capability ABCs (for isinstance checks / type annotations) ────────────────
  TextGenerationCapability
  StructuredGenerationCapability
  ToolCallingCapability
  ModerationCapability
  TranscriptionCapability
  SynthesisCapability

── Per-capability result models ─────────────────────────────────────────────
  ModerationResult      .label  .raw_text  .usage  .latency  …
  TranscriptionResult   .text   .language  .task   .latency  …
  SynthesisResult       .audio  .voice     .response_format  …

  Note: ChatClient, StructuredClient, and ToolClient all return the existing
  GroqResponse from the models package — no new result type needed.

── Registry helpers ─────────────────────────────────────────────────────────
  MODEL_CAPABILITIES    dict[model_id, set[Capability]]
  assert_capability()   raises CapabilityError early if model can't do it
  models_for()          list all models supporting a given capability
  CapabilityError       raised by assert_capability()

── Base ─────────────────────────────────────────────────────────────────────
  BaseGroqClient        base class for custom capability clients
"""

from clients.base import BaseGroqClient
from clients.capabilities import (
	ModerationCapability,
	StructuredGenerationCapability,
	SynthesisCapability,
	TextGenerationCapability,
	ToolCallingCapability,
	TranscriptionCapability,
	)
from clients.chat_client import ChatClient
from clients.moderation_client import ModerationClient, ModerationResult
from clients.registry import (assert_capability, Capability, CapabilityError, MODEL_CAPABILITIES, models_for)
from clients.structured_client import StructuredClient
from clients.synthesis_client import SynthesisClient, SynthesisResult
from clients.tool_client import ToolClient
from clients.transcription_client import TranscriptionClient, TranscriptionResult


__all__ = [
	# ── Concrete clients ──────────────────────────────────────────────
	"ChatClient",
	"StructuredClient",
	"ToolClient",
	"ModerationClient",
	"TranscriptionClient",
	"SynthesisClient",
	# ── Capability ABCs ───────────────────────────────────────────────
	"TextGenerationCapability",
	"StructuredGenerationCapability",
	"ToolCallingCapability",
	"ModerationCapability",
	"TranscriptionCapability",
	"SynthesisCapability",
	# ── Result models ─────────────────────────────────────────────────
	"ModerationResult",
	"TranscriptionResult",
	"SynthesisResult",
	# ── Registry ──────────────────────────────────────────────────────
	"MODEL_CAPABILITIES",
	"Capability",
	"CapabilityError",
	"assert_capability",
	"models_for",
	# ── Base ──────────────────────────────────────────────────────────
	"BaseGroqClient",
	]
