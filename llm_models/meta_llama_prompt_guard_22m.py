"""`meta-llama/llama-prompt-guard-2-22m` rate-limit configuration — Groq Free Plan defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llm_models.base import ModelRateLimitConfig


@dataclass
class LlamaPromptGuard222MConfig(ModelRateLimitConfig):
	"""Free Plan rate limits for ``meta-llama/llama-prompt-guard-2-22m``.

	Env-var overrides (all optional; class defaults apply when unset):
			GROQ_MODEL_PROMPT_GUARD_22M_RPM   – requests  / minute  (default: 30)
			GROQ_MODEL_PROMPT_GUARD_22M_RPD   – requests  / day     (default: 14,400)
			GROQ_MODEL_PROMPT_GUARD_22M_TPM   – tokens    / minute  (default: 15,000)
			GROQ_MODEL_PROMPT_GUARD_22M_TPD   – tokens    / day     (default: 500,000)
			GROQ_MODEL_PROMPT_GUARD_22M_ITPM  – in-tok    / minute  (default: None — only if split limits apply)
			GROQ_MODEL_PROMPT_GUARD_22M_OTPM  – out-tok   / minute  (default: None — only if split limits apply)
	"""

	# ------------------------------------------------------------------ identity
	model_id: str = "meta-llama/llama-prompt-guard-2-22m"
	plan: str = "free"

	# ------------------------------------------------------------------ rate limits
	requests_per_minute: int | None = 30  # requests per minute
	requests_per_day: int | None = 14_400  # requests per day
	tokens_per_minute: int | None = 15_000  # tokens per minute
	tokens_per_day: int | None = 500_000  # tokens per day

	# ------------------------------------------------------------------ env-var prefix
	_ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_PROMPT_GUARD_22M"

	context_window: int | None = 8_192
	max_output_tokens: int | None = 1_024
