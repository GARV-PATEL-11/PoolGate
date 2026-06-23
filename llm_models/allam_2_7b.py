"""`allam-2-7b` rate-limit configuration — Groq Free Plan defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llm_models.base import ModelRateLimitConfig


@dataclass
class Allam27BConfig(ModelRateLimitConfig):
    """Free Plan rate limits for ``allam-2-7b``.

    Env-var overrides (all optional; class defaults apply when unset):
                    GROQ_MODEL_ALLAM_2_7B_REQUESTS_PER_MINUTE – requests / minute  (default: 30)
                    GROQ_MODEL_ALLAM_2_7B_REQUESTS_PER_DAY    – requests / day         (default: 7,000)
                    GROQ_MODEL_ALLAM_2_7B_TOKENS_PER_MINUTE   – tokens    / minute   (default: 6,000)
                    GROQ_MODEL_ALLAM_2_7B_TOKENS_PER_DAY      – tokens    / day         (default: 500,000)
                    GROQ_MODEL_ALLAM_2_7B_INPUT_TOKENS_PER_MINUTE  – input tokens  / minute  (default: None — only if split limits apply)
                    GROQ_MODEL_ALLAM_2_7B_OUTPUT_TOKENS_PER_MINUTE – output tokens / minute  (default: None — only if split limits apply)
    """

    # ------------------------------------------------------------------ identity
    model_id: str = "allam-2-7b"
    plan: str = "free"

    # ------------------------------------------------------------------ rate limits
    requests_per_minute: int | None = 30  # requests per minute
    requests_per_day: int | None = 7_000  # requests per day
    tokens_per_minute: int | None = 6_000  # tokens per minute
    tokens_per_day: int | None = 500_000  # tokens per day

    # ------------------------------------------------------------------ env-var prefix
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_ALLAM_2_7B"

    context_window: int | None = 4_096
    max_output_tokens: int | None = 4_096
