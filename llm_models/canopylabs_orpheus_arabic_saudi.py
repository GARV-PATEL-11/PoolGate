"""`canopylabs/orpheus-arabic-saudi` rate-limit configuration — Groq Free Plan defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llm_models.base import ModelRateLimitConfig


@dataclass
class OrpheusArabicSaudiConfig(ModelRateLimitConfig):
    """Free Plan rate limits for ``canopylabs/orpheus-arabic-saudi``.

    Env-var overrides (all optional; class defaults apply when unset):
                    GROQ_MODEL_ORPHEUS_ARABIC_SAUDI_REQUESTS_PER_MINUTE – requests / minute  (default: 10)
                    GROQ_MODEL_ORPHEUS_ARABIC_SAUDI_REQUESTS_PER_DAY    – requests / day         (default: 100)
                    GROQ_MODEL_ORPHEUS_ARABIC_SAUDI_TOKENS_PER_MINUTE   – tokens    / minute   (default: 1,200)
                    GROQ_MODEL_ORPHEUS_ARABIC_SAUDI_TOKENS_PER_DAY      – tokens    / day         (default: 3,600)
                    GROQ_MODEL_ORPHEUS_ARABIC_SAUDI_INPUT_TOKENS_PER_MINUTE  – input tokens  / minute  (default: None — only if split limits apply)
                    GROQ_MODEL_ORPHEUS_ARABIC_SAUDI_OUTPUT_TOKENS_PER_MINUTE – output tokens / minute  (default: None — only if split limits apply)
    """

    # ------------------------------------------------------------------ identity
    model_id: str = "canopylabs/orpheus-arabic-saudi"
    plan: str = "free"

    # ------------------------------------------------------------------ rate limits
    requests_per_minute: int | None = 10  # requests per minute
    requests_per_day: int | None = 100  # requests per day
    tokens_per_minute: int | None = 1_200  # tokens per minute
    tokens_per_day: int | None = 3_600  # tokens per day

    # ------------------------------------------------------------------ env-var prefix
    _ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_ORPHEUS_ARABIC_SAUDI"
