"""Google Gemini model rate-limit configurations (Free Tier)."""

from __future__ import annotations

from dataclasses import dataclass, field

from poolgate.exceptions.request import UnknownModelError


@dataclass
class GeminiModelRateLimitConfig:
    """Per-model Gemini rate-limit configuration."""

    model_id: str = ""
    requests_per_minute: int = 5
    tokens_per_minute: int = 250_000
    requests_per_day: int = 20
    plan: str = "free"
    context_window: int | None = None
    max_output_tokens: int | None = None
    supports_structured_output: bool = True
    supports_tool_calling: bool = True


Gemini25FlashConfig = GeminiModelRateLimitConfig(
    model_id="gemini-2.5-flash",
    requests_per_minute=5,
    tokens_per_minute=250_000,
    requests_per_day=20,
    context_window=1_000_000,
    max_output_tokens=8192,
)

Gemini35FlashConfig = GeminiModelRateLimitConfig(
    model_id="gemini-3.5-flash",
    requests_per_minute=5,
    tokens_per_minute=250_000,
    requests_per_day=20,
    context_window=1_000_000,
    max_output_tokens=8192,
)

GEMINI_MODEL_REGISTRY: dict[str, GeminiModelRateLimitConfig] = {
    "gemini-2.5-flash": Gemini25FlashConfig,
    "gemini-3.5-flash": Gemini35FlashConfig,
}

GEMINI_MODEL_CAPABILITIES: dict[str, set[str]] = {
    "gemini-2.5-flash": {"chat", "structured", "tools"},
    "gemini-3.5-flash": {"chat", "structured", "tools"},
}


def get_gemini_model_config(model_id: str) -> GeminiModelRateLimitConfig:
    config = GEMINI_MODEL_REGISTRY.get(model_id)
    if config is None:
        raise UnknownModelError(
            f"Model '{model_id}' is not registered. Known Gemini models: {sorted(GEMINI_MODEL_REGISTRY)}",
            model_id=model_id,
        )
    return config


def assert_gemini_capability(model_id: str, capability: str) -> None:
    from poolgate.exceptions.request import CapabilityError

    caps = GEMINI_MODEL_CAPABILITIES.get(model_id)
    if caps is None:
        raise CapabilityError(
            f"Model '{model_id}' is not a registered Gemini model. Known: {sorted(GEMINI_MODEL_CAPABILITIES)}",
            model_id=model_id,
            capability=capability,
            supported_capabilities=[],
        )
    if capability not in caps:
        raise CapabilityError(
            f"Gemini model '{model_id}' does not support '{capability}'. Supported: {sorted(caps)}",
            model_id=model_id,
            capability=capability,
            supported_capabilities=sorted(caps),
        )
