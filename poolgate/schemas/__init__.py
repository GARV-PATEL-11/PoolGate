"""Full flat re-export of all PoolGate schemas for backwards-compatible imports."""

from poolgate.schemas.common import FinishReason, Metadata, Region, UTCTimestamp, utc_now, utcnow
from poolgate.schemas.common.context import RequestContext, RequestOptions, RequestType, Session
from poolgate.schemas.common.envelope import (
    CapabilityRequest,
    CapabilityResponse,
    PoolGateRequest,
    PoolGateResponse,
    PoolGetRequest,
    PoolGetResponse,
)
from poolgate.schemas.common.keys import AccountIdentity, APIKey, APIKeyIdentity
from poolgate.schemas.common.model_info import Capability, CapabilitySet, ModelCapabilities, ModelInfo
from poolgate.schemas.common.ops import ErrorResponse, HealthStatus, RetryPolicy
from poolgate.schemas.common.runtime import (
    APIKeyStatus,
    BatchResult,
    BatchSummary,
    GroqResponse,
    RequestConfig,
    RuntimeChatMessage,
)
from poolgate.schemas.requests.chat import ChatMessage, ChatRequest, ToolCall, ToolDefinition
from poolgate.schemas.requests.moderation import ModerationRequest, ModerationResponse
from poolgate.schemas.requests.structured import StructuredRequest, StructuredResponse
from poolgate.schemas.requests.synthesis import SynthesisRequest, SynthesisResponse
from poolgate.schemas.requests.transcription import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranslationRequest,
    TranslationResponse,
)
from poolgate.schemas.responses.usage import QuotaStatus, RequestUsage, TokenUsage

__all__ = [
    # common primitives
    "FinishReason",
    "Metadata",
    "Region",
    "UTCTimestamp",
    "utcnow",
    "utc_now",
    # context
    "RequestContext",
    "RequestOptions",
    "RequestType",
    "Session",
    # envelope
    "CapabilityRequest",
    "CapabilityResponse",
    "PoolGateRequest",
    "PoolGateResponse",
    "PoolGetRequest",
    "PoolGetResponse",
    # keys
    "AccountIdentity",
    "APIKey",
    "APIKeyIdentity",
    # model info
    "Capability",
    "CapabilitySet",
    "ModelCapabilities",
    "ModelInfo",
    # ops
    "ErrorResponse",
    "HealthStatus",
    "RetryPolicy",
    # runtime
    "APIKeyStatus",
    "BatchResult",
    "BatchSummary",
    "GroqResponse",
    "RequestConfig",
    "RuntimeChatMessage",
    # requests
    "ChatMessage",
    "ChatRequest",
    "ToolCall",
    "ToolDefinition",
    "ModerationRequest",
    "ModerationResponse",
    "StructuredRequest",
    "StructuredResponse",
    "SynthesisRequest",
    "SynthesisResponse",
    "TranscriptionRequest",
    "TranscriptionResponse",
    "TranslationRequest",
    "TranslationResponse",
    # responses
    "QuotaStatus",
    "RequestUsage",
    "TokenUsage",
]
