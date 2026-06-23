"""`llama-3.1-8b-instant` rate-limit configuration — Groq Free Plan defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llm_models.base import ModelRateLimitConfig


@dataclass
class Llama318BInstantConfig(ModelRateLimitConfig):
	"""Free Plan rate limits for ``llama-3.1-8b-instant``.

	Env-var overrides (all optional; class defaults apply when unset):
			GROQ_MODEL_LLAMA_31_8B_INSTANT_REQUESTS_PER_MINUTE – requests / minute  (default: 30)
			GROQ_MODEL_LLAMA_31_8B_INSTANT_REQUESTS_PER_DAY    – requests / day         (default: 14,400)
			GROQ_MODEL_LLAMA_31_8B_INSTANT_TOKENS_PER_MINUTE   – tokens    / minute   (default: 6,000)
			GROQ_MODEL_LLAMA_31_8B_INSTANT_TOKENS_PER_DAY      – tokens    / day         (default: 500,000)
			GROQ_MODEL_LLAMA_31_8B_INSTANT_INPUT_TOKENS_PER_MINUTE  – input tokens  / minute  (default: None — only if split limits apply)
			GROQ_MODEL_LLAMA_31_8B_INSTANT_OUTPUT_TOKENS_PER_MINUTE – output tokens / minute  (default: None — only if split limits apply)
	"""

	# ------------------------------------------------------------------ identity
	model_id: str = "llama-3.1-8b-instant"
	plan: str = "free"

	# ------------------------------------------------------------------ rate limits
	requests_per_minute: int | None = 30  # requests per minute
	requests_per_day: int | None = 14_400  # requests per day
	tokens_per_minute: int | None = 6_000  # tokens per minute
	tokens_per_day: int | None = 500_000  # tokens per day

	# ------------------------------------------------------------------ env-var prefix
	_ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_LLAMA_31_8B_INSTANT"

	context_window: int | None = 131_072
	max_output_tokens: int | None = 8_192
