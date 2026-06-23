"""`llama-3.3-70b-versatile` rate-limit configuration — Groq Free Plan defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llm_models.base import ModelRateLimitConfig


@dataclass
class Llama3370BVersatileConfig(ModelRateLimitConfig):
    """Free Plan rate limits for ``llama-3.3-70b-versatile``.

    Env-var overrides (all optional; class defaults apply when unset):
                    GROQ_MODEL_LLAMA_33_70B_VERSATILE_REQUESTS_PER_MINUTE – requests / minute  (default: 30)
                    GROQ_MODEL_LLAMA_33_70B_VERSATILE_REQUESTS_PER_DAY    – requests / day         (default: 1,000)
                    GROQ_MODEL_LLAMA_33_70B_VERSATILE_TOKENS_PER_MINUTE   – tokens    / minute   (default: 12,000)
                    GROQ_MODEL_LLAMA_33_70B_VERSATILE_TOKENS_PER_DAY      – tokens    / day         (default: 100,000)
                    GROQ_MODEL_LLAMA_33_70B_VERSATILE_INPUT_TOKENS_PER_MINUTE  – input tokens  / minute  (default: None — only if split limits apply)
                    GROQ_MODEL_LLAMA_33_70B_VERSATILE_OUTPUT_TOKENS_PER_MINUTE – output tokens / minute  (default: None — only if split limits apply)
    """

    # ------------------------------------------------------------------ identity
    model_id: str = "llama-3.3-70b-versatile"
    plan: str = "free"

    # ------------------------------------------------------------------ rate limits
    requests_per_minute: int | None = 30  # requests per minute
    requests_per_day: int | None = 1_000  # requests per day
    tokens_per_minute: int | None = 12_000  # tokens per minute
    tokens_per_day: int | None = 100_000  # tokens per day

    # ------------------------------------------------------------------ env-var prefix
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_LLAMA_33_70B_VERSATILE"

    context_window: int | None = 131_072
    max_output_tokens: int | None = 32_768
