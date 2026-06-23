"""
schemas/__init__.py — Public surface of the PoolGate schemas package.

Import everything from here. This package defines PoolGate's service-boundary
contracts — what's validated at the edges (an eventual FastAPI layer, internal
queue messages between service.py and schedulers/, etc.) — as opposed to the
low-level models.RequestConfig / models.TokenUsage / models.FinishReason used
internally by clients/ to talk to the Groq SDK.

A few names are intentionally duplicated against models/ and clients/registry.py
(TokenUsage, FinishReason, Capability) — see the module docstrings in common.py,
usage.py, and model_info.py for the reasoning and a note on consolidating later.

── Common ────────────────────────────────────────────────────────────────────
  Metadata, FinishReason, utcnow

── Model registry ──────────────────────────────────────────────────────────
  ModelInfo, ModelCapabilities, CapabilitySet, Capability

── Identity ─────────────────────────────────────────────────────────────────
  APIKey

── Chat + tool calling ──────────────────────────────────────────────────────
  ChatMessage, ChatRequest, ChatResponse, ToolDefinition, ToolCall

── Structured generation ────────────────────────────────────────────────────
  StructuredRequest, StructuredResponse

── Moderation ───────────────────────────────────────────────────────────────
  ModerationRequest, ModerationResponse

── Transcription ────────────────────────────────────────────────────────────
  TranscriptionRequest, TranscriptionResponse

── Synthesis ────────────────────────────────────────────────────────────────
  SynthesisRequest, SynthesisResponse

── Request lifecycle ────────────────────────────────────────────────────────
  RequestContext, RequestOptions, Session

── Usage / quota ────────────────────────────────────────────────────────────
  TokenUsage, RequestUsage, QuotaStatus

── Ops ──────────────────────────────────────────────────────────────────────
  HealthStatus, RetryPolicy, ErrorResponse

── Envelope ─────────────────────────────────────────────────────────────────
  PoolGateRequest, PoolGateResponse, CapabilityRequest, CapabilityResponse
"""

from schemas.chat import ChatMessage, ChatRequest, ChatResponse, ToolCall, ToolDefinition
from schemas.common import FinishReason, Metadata, Region, utc_now, utcnow, UTCTimestamp
from schemas.context import RequestContext, RequestOptions, RequestType, Session
from schemas.envelope import (
	CapabilityRequest,
	CapabilityResponse,
	PoolGateRequest,
	PoolGateResponse,
	PoolGetRequest,
	PoolGetResponse,
	)
from schemas.keys import AccountIdentity, APIKey, APIKeyIdentity
from schemas.model_info import Capability, CapabilitySet, ModelCapabilities, ModelInfo
from schemas.moderation import ModerationRequest, ModerationResponse
from schemas.ops import ErrorResponse, HealthStatus, RetryPolicy
from schemas.runtime import (
	APIKeyStatus,
	BatchResult,
	BatchSummary,
	GroqResponse,
	RequestConfig,
	RuntimeChatMessage,
	)
from schemas.structured import StructuredRequest, StructuredResponse
from schemas.synthesis import SynthesisRequest, SynthesisResponse
from schemas.transcription import (
	TranscriptionRequest,
	TranscriptionResponse,
	TranslationRequest,
	TranslationResponse,
	)
from schemas.usage import QuotaStatus, RequestUsage, TokenUsage


__all__ = [
	# Common
	"Metadata",
	"FinishReason",
	"Region",
	"UTCTimestamp",
	"utcnow",
	"utc_now",
	# Model registry
	"ModelInfo",
	"ModelCapabilities",
	"CapabilitySet",
	"Capability",
	"APIKeyStatus",
	# Identity
	"APIKey",
	"APIKeyIdentity",
	"AccountIdentity",
	# Chat + tools
	"ChatMessage",
	"RuntimeChatMessage",
	"ChatRequest",
	"ChatResponse",
	"ToolDefinition",
	"ToolCall",
	# Structured
	"StructuredRequest",
	"StructuredResponse",
	# Moderation
	"ModerationRequest",
	"ModerationResponse",
	# Transcription
	"TranscriptionRequest",
	"TranscriptionResponse",
	"TranslationRequest",
	"TranslationResponse",
	# Synthesis
	"SynthesisRequest",
	"SynthesisResponse",
	# Request lifecycle
	"RequestContext",
	"RequestOptions",
	"Session",
	"RequestType",
	# Usage / quota
	"TokenUsage",
	"RequestConfig",
	"GroqResponse",
	"BatchResult",
	"BatchSummary",
	"RequestUsage",
	"QuotaStatus",
	# Ops
	"HealthStatus",
	"RetryPolicy",
	"ErrorResponse",
	# Envelope
	"PoolGateRequest",
	"PoolGateResponse",
	"PoolGetRequest",
	"PoolGetResponse",
	"CapabilityRequest",
	"CapabilityResponse",
	]
