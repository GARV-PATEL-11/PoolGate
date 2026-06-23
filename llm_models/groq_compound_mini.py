"""`groq/compound-mini` rate-limit configuration — Groq Free Plan defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llm_models.base import ModelRateLimitConfig


@dataclass
class CompoundMiniConfig(ModelRateLimitConfig):
    """Free Plan rate limits for ``groq/compound-mini``.

    Env-var overrides (all optional; class defaults apply when unset):
                    GROQ_MODEL_COMPOUND_MINI_REQUESTS_PER_MINUTE – requests / minute  (default: 30)
                    GROQ_MODEL_COMPOUND_MINI_REQUESTS_PER_DAY    – requests / day         (default: 250)
                    GROQ_MODEL_COMPOUND_MINI_TOKENS_PER_MINUTE   – tokens    / minute   (default: 70,000)
                    GROQ_MODEL_COMPOUND_MINI_TOKENS_PER_DAY      – tokens    / day         (default: None)
                    GROQ_MODEL_COMPOUND_MINI_INPUT_TOKENS_PER_MINUTE  – input tokens  / minute  (default: None — only if split limits apply)
                    GROQ_MODEL_COMPOUND_MINI_OUTPUT_TOKENS_PER_MINUTE – output tokens / minute  (default: None — only if split limits apply)
    """

    # ------------------------------------------------------------------ identity
    model_id: str = "groq/compound-mini"
    plan: str = "free"

    # ------------------------------------------------------------------ rate limits
    requests_per_minute: int | None = 30  # requests per minute
    requests_per_day: int | None = 250  # requests per day
    tokens_per_minute: int | None = 70_000  # tokens per minute
    tokens_per_day: int | None = None  # tokens per day

    # ------------------------------------------------------------------ env-var prefix
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_COMPOUND_MINI"
