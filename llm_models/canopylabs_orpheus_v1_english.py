"""`canopylabs/orpheus-v1-english` rate-limit configuration — Groq Free Plan defaults."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llm_models.base import ModelRateLimitConfig


@dataclass
class OrpheusV1EnglishConfig(ModelRateLimitConfig):
	"""Free Plan rate limits for ``canopylabs/orpheus-v1-english``.

	Env-var overrides (all optional; class defaults apply when unset):
		GROQ_MODEL_ORPHEUS_V1_ENGLISH_RPM   – requests  / minute  (default: 10)
		GROQ_MODEL_ORPHEUS_V1_ENGLISH_RPD   – requests  / day     (default: 100)
		GROQ_MODEL_ORPHEUS_V1_ENGLISH_TPM   – tokens    / minute  (default: 1,200)
		GROQ_MODEL_ORPHEUS_V1_ENGLISH_TPD   – tokens    / day     (default: 3,600)
		GROQ_MODEL_ORPHEUS_V1_ENGLISH_ITPM  – in-tok    / minute  (default: None — only if split limits apply)
		GROQ_MODEL_ORPHEUS_V1_ENGLISH_OTPM  – out-tok   / minute  (default: None — only if split limits apply)
	"""

	# ------------------------------------------------------------------ identity
	model_id: str = "canopylabs/orpheus-v1-english"
	plan: str = "free"

	# ------------------------------------------------------------------ rate limits
	requests_per_minute: int | None = 10  # requests per minute
	requests_per_day: int | None = 100  # requests per day
	tokens_per_minute: int | None = 1_200  # tokens per minute
	tokens_per_day: int | None = 3_600  # tokens per day

	# ------------------------------------------------------------------ env-var prefix
	_ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_ORPHEUS_V1_ENGLISH"
