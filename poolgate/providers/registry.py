"""Model → capability mapping. Capability is canonical in schemas.common.model_info."""

from __future__ import annotations

from poolgate.exceptions.request import CapabilityError
from poolgate.schemas.common.model_info import Capability

MODEL_CAPABILITIES: dict[str, set[Capability]] = {
    "allam-2-7b": {"chat"},
    "llama-3.1-8b-instant": {"chat", "structured", "tools"},
    "llama-3.3-70b-versatile": {"chat", "structured", "tools"},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"chat", "structured", "tools"},
    "openai/gpt-oss-20b": {"chat", "structured", "tools"},
    "openai/gpt-oss-120b": {"chat", "structured", "tools"},
    "qwen/qwen3-32b": {"chat", "structured", "tools"},
    "qwen/qwen3.6-27b": {"chat", "structured", "tools"},
    "groq/compound": {"chat", "tools"},
    "groq/compound-mini": {"chat", "tools"},
    "meta-llama/llama-prompt-guard-2-22m": {"moderation"},
    "meta-llama/llama-prompt-guard-2-86m": {"moderation"},
    "openai/gpt-oss-safeguard-20b": {"moderation"},
    "whisper-large-v3": {"transcription"},
    "whisper-large-v3-turbo": {"transcription"},
    "canopylabs/orpheus-arabic-saudi": {"synthesis"},
    "canopylabs/orpheus-v1-english": {"synthesis"},
}


def assert_capability(model: str, capability: Capability) -> None:
    caps = MODEL_CAPABILITIES.get(model)
    if caps is None:
        raise CapabilityError(
            f"Model '{model}' is not registered in PoolGate. Known models: {sorted(MODEL_CAPABILITIES)}",
            model_id=model,
            capability=capability,
            supported_capabilities=[],
        )
    if capability not in caps:
        raise CapabilityError(
            f"Model '{model}' does not support '{capability}'. Supported: {sorted(caps)}",
            model_id=model,
            capability=capability,
            supported_capabilities=sorted(caps),
        )


def models_for(capability: Capability) -> list[str]:
    return sorted(m for m, caps in MODEL_CAPABILITIES.items() if capability in caps)
