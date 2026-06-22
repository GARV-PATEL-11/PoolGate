"""`meta-llama/llama-4-scout-17b-16e-instruct` rate-limit configuration — Groq Free Plan defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llm_models.base import ModelRateLimitConfig


@dataclass
class Llama4Scout17BConfig(ModelRateLimitConfig):
    """Free Plan rate limits for ``meta-llama/llama-4-scout-17b-16e-instruct``.

    Env-var overrides (all optional; class defaults apply when unset):
            GROQ_MODEL_LLAMA4_SCOUT_17B_RPM   – requests  / minute  (default: 30)
            GROQ_MODEL_LLAMA4_SCOUT_17B_RPD   – requests  / day     (default: 1,000)
            GROQ_MODEL_LLAMA4_SCOUT_17B_TPM   – tokens    / minute  (default: 30,000)
            GROQ_MODEL_LLAMA4_SCOUT_17B_TPD   – tokens    / day     (default: 500,000)
            GROQ_MODEL_LLAMA4_SCOUT_17B_ITPM  – in-tok    / minute  (default: None — only if split limits apply)
            GROQ_MODEL_LLAMA4_SCOUT_17B_OTPM  – out-tok   / minute  (default: None — only if split limits apply)
    """

    # ------------------------------------------------------------------ identity
    model_id: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    plan: str = "free"

    # ------------------------------------------------------------------ rate limits
    requests_per_minute: int | None = 30  # requests per minute
    requests_per_day: int | None = 1_000  # requests per day
    tokens_per_minute: int | None = 30_000  # tokens per minute
    tokens_per_day: int | None = 500_000  # tokens per day

    # ------------------------------------------------------------------ env-var prefix
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_LLAMA4_SCOUT_17B"

    context_window: int | None = 10_000_000
    max_output_tokens: int | None = 8_192
