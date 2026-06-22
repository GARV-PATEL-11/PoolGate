"""`openai/gpt-oss-20b` rate-limit configuration — Groq Free Plan defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llm_models.base import ModelRateLimitConfig


@dataclass
class GptOss20BConfig(ModelRateLimitConfig):
    """Free Plan rate limits for ``openai/gpt-oss-20b``.

    Env-var overrides (all optional; class defaults apply when unset):
            GROQ_MODEL_GPT_OSS_20B_RPM   – requests  / minute  (default: 30)
            GROQ_MODEL_GPT_OSS_20B_RPD   – requests  / day     (default: 1,000)
            GROQ_MODEL_GPT_OSS_20B_TPM   – tokens    / minute  (default: 8,000)
            GROQ_MODEL_GPT_OSS_20B_TPD   – tokens    / day     (default: 200,000)
            GROQ_MODEL_GPT_OSS_20B_ITPM  – in-tok    / minute  (default: None — only if split limits apply)
            GROQ_MODEL_GPT_OSS_20B_OTPM  – out-tok   / minute  (default: None — only if split limits apply)
    """

    # ------------------------------------------------------------------ identity
    model_id: str = "openai/gpt-oss-20b"
    plan: str = "free"

    # ------------------------------------------------------------------ rate limits
    requests_per_minute: int | None = 30  # requests per minute
    requests_per_day: int | None = 1_000  # requests per day
    tokens_per_minute: int | None = 8_000  # tokens per minute
    tokens_per_day: int | None = 200_000  # tokens per day

    # ------------------------------------------------------------------ env-var prefix
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_GPT_OSS_20B"

    context_window: int | None = 131_072
    max_output_tokens: int | None = 8_192
