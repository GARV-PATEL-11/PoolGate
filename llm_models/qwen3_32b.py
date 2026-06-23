"""`qwen/qwen3-32b` rate-limit configuration — Groq Free Plan defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llm_models.base import ModelRateLimitConfig


@dataclass
class Qwen332BConfig(ModelRateLimitConfig):
    """Free Plan rate limits for ``qwen/qwen3-32b``.

    Env-var overrides (all optional; class defaults apply when unset):
                    GROQ_MODEL_QWEN3_32B_REQUESTS_PER_MINUTE – requests / minute  (default: 60)
                    GROQ_MODEL_QWEN3_32B_REQUESTS_PER_DAY    – requests / day         (default: 1,000)
                    GROQ_MODEL_QWEN3_32B_TOKENS_PER_MINUTE   – tokens    / minute   (default: 6,000)
                    GROQ_MODEL_QWEN3_32B_TOKENS_PER_DAY      – tokens    / day         (default: 500,000)
                    GROQ_MODEL_QWEN3_32B_INPUT_TOKENS_PER_MINUTE  – input tokens  / minute  (default: None — only if split limits apply)
                    GROQ_MODEL_QWEN3_32B_OUTPUT_TOKENS_PER_MINUTE – output tokens / minute  (default: None — only if split limits apply)
    """

    # ------------------------------------------------------------------ identity
    model_id: str = "qwen/qwen3-32b"
    plan: str = "free"

    # ------------------------------------------------------------------ rate limits
    requests_per_minute: int | None = 60  # requests per minute
    requests_per_day: int | None = 1_000  # requests per day
    tokens_per_minute: int | None = 6_000  # tokens per minute
    tokens_per_day: int | None = 500_000  # tokens per day

    # ------------------------------------------------------------------ env-var prefix
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_QWEN3_32B"

    context_window: int | None = 131_072
    max_output_tokens: int | None = 8_192
