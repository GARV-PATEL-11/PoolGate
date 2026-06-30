"""Consolidated Groq model rate-limit configurations (Groq Free Plan)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

from poolgate.exceptions.configuration import EnvironmentParseError, InvalidRateLimitConfigError
from poolgate.exceptions.request import UnknownModelError


@dataclass
class ModelRateLimitConfig:
    """Per-model rate-limit config — plan defaults, env-overridable at runtime."""

    requests_per_minute: int | None = None
    requests_per_day: int | None = None
    tokens_per_minute: int | None = None
    tokens_per_day: int | None = None
    audio_seconds_per_hour: int | None = None
    audio_seconds_per_day: int | None = None
    input_tokens_per_minute: int | None = None
    output_tokens_per_minute: int | None = None

    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL"

    model_id: str = ""
    plan: str = "free"
    context_window: int | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "requests_per_minute",
            "requests_per_day",
            "tokens_per_minute",
            "tokens_per_day",
            "audio_seconds_per_hour",
            "audio_seconds_per_day",
            "input_tokens_per_minute",
            "output_tokens_per_minute",
        ):
            val: int | None = getattr(self, name)
            if val is not None and val <= 0:
                raise InvalidRateLimitConfigError(
                    f"{self.__class__.__name__}.{name} must be a positive integer; got {val!r}.",
                    field=name,
                    value=val,
                )
        for name in ("context_window", "max_output_tokens"):
            val = getattr(self, name)
            if val is not None and val <= 0:
                raise InvalidRateLimitConfigError(
                    f"{self.__class__.__name__}.{name} must be a positive integer; got {val!r}.",
                    field=name,
                    value=val,
                )

    @property
    def is_audio_model(self) -> bool:
        return self.audio_seconds_per_hour is not None or self.audio_seconds_per_day is not None

    @property
    def is_text_model(self) -> bool:
        return self.tokens_per_minute is not None or self.tokens_per_day is not None

    @property
    def has_split_token_limits(self) -> bool:
        return self.input_tokens_per_minute is not None or self.output_tokens_per_minute is not None

    def active_limits(self) -> dict[str, int]:
        keys = (
            "requests_per_minute",
            "requests_per_day",
            "tokens_per_minute",
            "tokens_per_day",
            "audio_seconds_per_hour",
            "audio_seconds_per_day",
            "input_tokens_per_minute",
            "output_tokens_per_minute",
        )
        return {k: v for k in keys if (v := getattr(self, k)) is not None}

    def to_dict(self) -> dict[str, object]:
        metadata = {k: v for k in ("context_window", "max_output_tokens") if (v := getattr(self, k)) is not None}
        return {"model_id": self.model_id, "plan": self.plan, **metadata, **self.active_limits()}

    @staticmethod
    def _env_int(var: str, default: int | None) -> int | None:
        raw = os.environ.get(var)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise EnvironmentParseError(
                f"{var} must be an integer; got {raw!r}.",
                var_name=var,
                raw_value=raw,
                expected=int,
            ) from exc

    @classmethod
    def from_env(cls) -> ModelRateLimitConfig:
        defaults = cls()
        p = cls._ENV_PREFIX
        return cls(
            model_id=defaults.model_id,
            plan=defaults.plan,
            requests_per_minute=cls._env_int(f"{p}_REQUESTS_PER_MINUTE", defaults.requests_per_minute),
            requests_per_day=cls._env_int(f"{p}_REQUESTS_PER_DAY", defaults.requests_per_day),
            tokens_per_minute=cls._env_int(f"{p}_TOKENS_PER_MINUTE", defaults.tokens_per_minute),
            tokens_per_day=cls._env_int(f"{p}_TOKENS_PER_DAY", defaults.tokens_per_day),
            audio_seconds_per_hour=cls._env_int(f"{p}_AUDIO_SECONDS_PER_HOUR", defaults.audio_seconds_per_hour),
            audio_seconds_per_day=cls._env_int(f"{p}_AUDIO_SECONDS_PER_DAY", defaults.audio_seconds_per_day),
            input_tokens_per_minute=cls._env_int(f"{p}_INPUT_TOKENS_PER_MINUTE", defaults.input_tokens_per_minute),
            output_tokens_per_minute=cls._env_int(f"{p}_OUTPUT_TOKENS_PER_MINUTE", defaults.output_tokens_per_minute),
        )


# ---------------------------------------------------------------------------
# Concrete model configs
# ---------------------------------------------------------------------------


@dataclass
class Allam27BConfig(ModelRateLimitConfig):
    model_id: str = "allam-2-7b"
    plan: str = "free"
    requests_per_minute: int | None = 30
    requests_per_day: int | None = 7_000
    tokens_per_minute: int | None = 6_000
    tokens_per_day: int | None = 500_000
    context_window: int | None = 4_096
    max_output_tokens: int | None = 4_096
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_ALLAM_2_7B"


@dataclass
class OrpheusArabicSaudiConfig(ModelRateLimitConfig):
    model_id: str = "canopylabs/orpheus-arabic-saudi"
    plan: str = "free"
    requests_per_minute: int | None = 10
    requests_per_day: int | None = 100
    tokens_per_minute: int | None = 1_200
    tokens_per_day: int | None = 3_600
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_ORPHEUS_ARABIC_SAUDI"


@dataclass
class OrpheusV1EnglishConfig(ModelRateLimitConfig):
    model_id: str = "canopylabs/orpheus-v1-english"
    plan: str = "free"
    requests_per_minute: int | None = 10
    requests_per_day: int | None = 100
    tokens_per_minute: int | None = 1_200
    tokens_per_day: int | None = 3_600
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_ORPHEUS_V1_ENGLISH"


@dataclass
class CompoundConfig(ModelRateLimitConfig):
    model_id: str = "groq/compound"
    plan: str = "free"
    requests_per_minute: int | None = 30
    requests_per_day: int | None = 250
    tokens_per_minute: int | None = 70_000
    tokens_per_day: int | None = None
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_COMPOUND"


@dataclass
class CompoundMiniConfig(ModelRateLimitConfig):
    model_id: str = "groq/compound-mini"
    plan: str = "free"
    requests_per_minute: int | None = 30
    requests_per_day: int | None = 250
    tokens_per_minute: int | None = 70_000
    tokens_per_day: int | None = None
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_COMPOUND_MINI"


@dataclass
class Llama318BInstantConfig(ModelRateLimitConfig):
    model_id: str = "llama-3.1-8b-instant"
    plan: str = "free"
    requests_per_minute: int | None = 30
    requests_per_day: int | None = 14_400
    tokens_per_minute: int | None = 6_000
    tokens_per_day: int | None = 500_000
    context_window: int | None = 131_072
    max_output_tokens: int | None = 8_192
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_LLAMA_31_8B_INSTANT"


@dataclass
class Llama3370BVersatileConfig(ModelRateLimitConfig):
    model_id: str = "llama-3.3-70b-versatile"
    plan: str = "free"
    requests_per_minute: int | None = 30
    requests_per_day: int | None = 1_000
    tokens_per_minute: int | None = 12_000
    tokens_per_day: int | None = 100_000
    context_window: int | None = 131_072
    max_output_tokens: int | None = 32_768
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_LLAMA_33_70B_VERSATILE"


@dataclass
class Llama4Scout17BConfig(ModelRateLimitConfig):
    model_id: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    plan: str = "free"
    requests_per_minute: int | None = 30
    requests_per_day: int | None = 1_000
    tokens_per_minute: int | None = 30_000
    tokens_per_day: int | None = 500_000
    context_window: int | None = 10_000_000
    max_output_tokens: int | None = 8_192
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_LLAMA4_SCOUT_17B"


@dataclass
class LlamaPromptGuard222MConfig(ModelRateLimitConfig):
    model_id: str = "meta-llama/llama-prompt-guard-2-22m"
    plan: str = "free"
    requests_per_minute: int | None = 30
    requests_per_day: int | None = 14_400
    tokens_per_minute: int | None = 15_000
    tokens_per_day: int | None = 500_000
    context_window: int | None = 8_192
    max_output_tokens: int | None = 1_024
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_PROMPT_GUARD_22M"


@dataclass
class LlamaPromptGuard286MConfig(ModelRateLimitConfig):
    model_id: str = "meta-llama/llama-prompt-guard-2-86m"
    plan: str = "free"
    requests_per_minute: int | None = 30
    requests_per_day: int | None = 14_400
    tokens_per_minute: int | None = 15_000
    tokens_per_day: int | None = 500_000
    context_window: int | None = 8_192
    max_output_tokens: int | None = 1_024
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_PROMPT_GUARD_86M"


@dataclass
class GptOss120BConfig(ModelRateLimitConfig):
    model_id: str = "openai/gpt-oss-120b"
    plan: str = "free"
    requests_per_minute: int | None = 30
    requests_per_day: int | None = 1_000
    tokens_per_minute: int | None = 8_000
    tokens_per_day: int | None = 200_000
    context_window: int | None = 131_072
    max_output_tokens: int | None = 8_192
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_GPT_OSS_120B"


@dataclass
class GptOss20BConfig(ModelRateLimitConfig):
    model_id: str = "openai/gpt-oss-20b"
    plan: str = "free"
    requests_per_minute: int | None = 30
    requests_per_day: int | None = 1_000
    tokens_per_minute: int | None = 8_000
    tokens_per_day: int | None = 200_000
    context_window: int | None = 131_072
    max_output_tokens: int | None = 8_192
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_GPT_OSS_20B"


@dataclass
class GptOssSafeguard20BConfig(ModelRateLimitConfig):
    model_id: str = "openai/gpt-oss-safeguard-20b"
    plan: str = "free"
    requests_per_minute: int | None = 30
    requests_per_day: int | None = 1_000
    tokens_per_minute: int | None = 8_000
    tokens_per_day: int | None = 200_000
    context_window: int | None = 131_072
    max_output_tokens: int | None = 8_192
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_GPT_OSS_SAFEGUARD_20B"


@dataclass
class Qwen332BConfig(ModelRateLimitConfig):
    model_id: str = "qwen/qwen3-32b"
    plan: str = "free"
    requests_per_minute: int | None = 60
    requests_per_day: int | None = 1_000
    tokens_per_minute: int | None = 6_000
    tokens_per_day: int | None = 500_000
    context_window: int | None = 131_072
    max_output_tokens: int | None = 8_192
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_QWEN3_32B"


@dataclass
class Qwen3627BConfig(ModelRateLimitConfig):
    model_id: str = "qwen/qwen3.6-27b"
    plan: str = "free"
    requests_per_minute: int | None = 30
    requests_per_day: int | None = 1_000
    tokens_per_minute: int | None = 8_000
    tokens_per_day: int | None = 200_000
    context_window: int | None = 131_072
    max_output_tokens: int | None = 8_192
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_QWEN3_6_27B"


@dataclass
class WhisperLargeV3Config(ModelRateLimitConfig):
    model_id: str = "whisper-large-v3"
    plan: str = "free"
    requests_per_minute: int | None = 20
    requests_per_day: int | None = 2_000
    audio_seconds_per_hour: int | None = 7_200
    audio_seconds_per_day: int | None = 28_800
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_WHISPER_V3"


@dataclass
class WhisperLargeV3TurboConfig(ModelRateLimitConfig):
    model_id: str = "whisper-large-v3-turbo"
    plan: str = "free"
    requests_per_minute: int | None = 20
    requests_per_day: int | None = 2_000
    audio_seconds_per_hour: int | None = 7_200
    audio_seconds_per_day: int | None = 28_800
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_WHISPER_V3_TURBO"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[ModelRateLimitConfig]] = {
    "allam-2-7b": Allam27BConfig,
    "canopylabs/orpheus-arabic-saudi": OrpheusArabicSaudiConfig,
    "canopylabs/orpheus-v1-english": OrpheusV1EnglishConfig,
    "groq/compound": CompoundConfig,
    "groq/compound-mini": CompoundMiniConfig,
    "llama-3.1-8b-instant": Llama318BInstantConfig,
    "llama-3.3-70b-versatile": Llama3370BVersatileConfig,
    "meta-llama/llama-4-scout-17b-16e-instruct": Llama4Scout17BConfig,
    "meta-llama/llama-prompt-guard-2-22m": LlamaPromptGuard222MConfig,
    "meta-llama/llama-prompt-guard-2-86m": LlamaPromptGuard286MConfig,
    "openai/gpt-oss-120b": GptOss120BConfig,
    "openai/gpt-oss-20b": GptOss20BConfig,
    "openai/gpt-oss-safeguard-20b": GptOssSafeguard20BConfig,
    "qwen/qwen3-32b": Qwen332BConfig,
    "qwen/qwen3.6-27b": Qwen3627BConfig,
    "whisper-large-v3": WhisperLargeV3Config,
    "whisper-large-v3-turbo": WhisperLargeV3TurboConfig,
}


def get_model_config(model_id: str) -> ModelRateLimitConfig:
    """Return an env-initialised ModelRateLimitConfig for model_id."""
    if model_id not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY)
        raise UnknownModelError(
            f"Unknown model {model_id!r}. Registered models: {available}",
            model_id=model_id,
            available_models=available,
        )
    return MODEL_REGISTRY[model_id].from_env()
