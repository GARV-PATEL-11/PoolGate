"""`whisper-large-v3` rate-limit configuration — Groq Free Plan defaults."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from llm_models.base import ModelRateLimitConfig


@dataclass
class WhisperLargeV3Config(ModelRateLimitConfig):
	"""Free Plan rate limits for ``whisper-large-v3`` (audio transcription).

	Env-var overrides (all optional; class defaults apply when unset):
		GROQ_MODEL_WHISPER_V3_RPM   – requests       / minute  (default: 20)
		GROQ_MODEL_WHISPER_V3_RPD   – requests       / day     (default: 2,000)
		GROQ_MODEL_WHISPER_V3_ASH   – audio seconds  / hour    (default: 7,200)
		GROQ_MODEL_WHISPER_V3_ASD   – audio seconds  / day     (default: 28,800)
	"""

	# ------------------------------------------------------------------ identity
	model_id: str = "whisper-large-v3"
	plan: str = "free"

	# ------------------------------------------------------------------ rate limits
	requests_per_minute: int | None = 20  # requests per minute
	requests_per_day: int | None = 2_000  # requests per day
	audio_seconds_per_hour: int | None = 7_200  # audio seconds per hour
	audio_seconds_per_day: int | None = 28_800  # audio seconds per day

	# ------------------------------------------------------------------ env-var prefix
	_ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_WHISPER_V3"
