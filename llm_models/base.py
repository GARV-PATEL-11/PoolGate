"""Base model rate-limit configuration for the Groq pool.

Every Groq model has a distinct rate-limit envelope that varies by plan.
Subclass :class:`ModelRateLimitConfig` per model, set class-level defaults
and ``_ENV_PREFIX``.  ``SubClass.from_env()`` then produces an instance that
respects any run-time environment-variable overrides — no additional code
required in the subclass.

Rate-limit field reference (Groq API docs):
    requests_per_minute   requests  per minute
    requests_per_day   requests  per day
    tokens_per_minute   tokens    per minute (combined in+out)
    tokens_per_day   tokens    per day
    audio_seconds_per_hour   audio seconds per hour
    audio_seconds_per_day   audio seconds per day
    input_tokens_per_minute  input  tokens per minute  (split-limit plans only)
    output_tokens_per_minute  output tokens per minute  (split-limit plans only)

Cached tokens do not count toward any rate limit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

from exceptions.configuration import EnvironmentParseError, InvalidRateLimitConfigError


@dataclass
class ModelRateLimitConfig:
	"""Per-model rate-limit config — plan defaults, env-overridable at runtime.

	Subclass pattern::

			@dataclass
			class Llama3370BVersatileConfig(ModelRateLimitConfig):
					model_id: str       = "llama-3.3-70b-versatile"
					plan:     str       = "free"
					requests_per_minute:  int | None = 30
					requests_per_day:  int | None = 1_000
					tokens_per_minute:  int | None = 12_000
					tokens_per_day:  int | None = 100_000
					_ENV_PREFIX: ClassVar[str] = "GROQ_MODEL_LLAMA_33_70B_VERSATILE"

	Construction::

			cfg = Llama3370BVersatileConfig.from_env()
			# GROQ_MODEL_LLAMA_33_70B_VERSATILE_RPM env var overrides 30 if set.

	Limits that do not apply to a model (e.g. tokens_per_minute / tokens_per_day for audio-only
	Whisper llm_models) should remain ``None`` in the subclass; they are
	excluded from :meth:`active_limits` and :meth:`to_dict`.
	"""

	# ------------------------------------------------------------------ limits (None = not applicable for this model)
	requests_per_minute: int | None = None  # requests per minute
	requests_per_day: int | None = None  # requests per day
	tokens_per_minute: int | None = None  # tokens per minute (combined in + out)
	tokens_per_day: int | None = None  # tokens per day
	audio_seconds_per_hour: int | None = None  # audio seconds per hour
	audio_seconds_per_day: int | None = None  # audio seconds per day
	input_tokens_per_minute: int | None = None  # input  tokens per minute  (split limits)
	output_tokens_per_minute: int | None = None  # output tokens per minute  (split limits)

	# ------------------------------------------------------------------ class var — NOT a dataclass field
	_ENV_PREFIX: ClassVar[str] = "GROQ_MODEL"

	# ------------------------------------------------------------------ identity
	model_id: str = ""
	plan: str = "free"  # "free" | "developer"
	context_window: int | None = None
	max_output_tokens: int | None = None

	# ------------------------------------------------------------------ validation
	def __post_init__(self) -> None:
		for name in (
				"requests_per_minute",
				"requests_per_day",
				"tokens_per_minute",
				"tokens_per_day",
				"audio_seconds_per_hour",
				"audio_seconds_per_day",
				"input_tokens_per_minute",
				"output_tokens_per_minute",
				):
			val: int | None = getattr(self, name)
			if val is not None and val <= 0:
				raise InvalidRateLimitConfigError(
					f"{self.__class__.__name__}.{name} must be a positive integer; got {val!r}.",
					field=name,
					value=val,
					)
		for name in ("context_window", "max_output_tokens"):
			val = getattr(self, name)
			if val is not None and val <= 0:
				raise InvalidRateLimitConfigError(
					f"{self.__class__.__name__}.{name} must be a positive integer; got {val!r}.",
					field=name,
					value=val,
					)

	# ------------------------------------------------------------------ derived properties
	@property
	def is_audio_model(self) -> bool:
		"""``True`` when this model has audio-second (audio_seconds_per_hour / audio_seconds_per_day) limits."""
		return self.audio_seconds_per_hour is not None or self.audio_seconds_per_day is not None

	@property
	def is_text_model(self) -> bool:
		"""``True`` when this model has token (tokens_per_minute / tokens_per_day) limits."""
		return self.tokens_per_minute is not None or self.tokens_per_day is not None

	@property
	def has_split_token_limits(self) -> bool:
		"""``True`` when separate input_tokens_per_minute / output_tokens_per_minute limits are active."""
		return self.input_tokens_per_minute is not None or self.output_tokens_per_minute is not None

	# ------------------------------------------------------------------ serialisation
	def active_limits(self) -> dict[str, int]:
		"""Return only the limits that are set (non-``None``)."""
		keys = (
			"requests_per_minute",
			"requests_per_day",
			"tokens_per_minute",
			"tokens_per_day",
			"audio_seconds_per_hour",
			"audio_seconds_per_day",
			"input_tokens_per_minute",
			"output_tokens_per_minute",
			)
		return {k: v for k in keys if (v := getattr(self, k)) is not None}

	def to_dict(self) -> dict[str, object]:
		"""Serialise to a plain ``dict`` (suitable for JSON / logging)."""
		metadata = {
			k: v
			for k in ("context_window", "max_output_tokens")
			if (v := getattr(self, k)) is not None
			}
		return {"model_id": self.model_id, "plan": self.plan, **metadata, **self.active_limits()}

	# ------------------------------------------------------------------ env helper
	@staticmethod
	def _env_int(var: str, default: int | None) -> int | None:
		"""Read an integer env var; fall back to *default* if unset."""
		raw = os.environ.get(var)
		if raw is None:
			return default
		try:
			return int(raw)
		except ValueError as exc:
			raise EnvironmentParseError(
				f"{var} must be an integer; got {raw!r}.",
				var_name=var,
				raw_value=raw,
				expected=int,
				) from exc

	# ------------------------------------------------------------------ factory
	@classmethod
	def from_env(cls) -> ModelRateLimitConfig:
		"""Build an instance with environment-variable overrides applied.

		For each limit field ``<_ENV_PREFIX>_<FIELD>`` wins when set in the
		environment; otherwise the class-level default is used.  This method
		works generically for every subclass without needing to be overridden.

		Example::

				GROQ_MODEL_LLAMA_33_70B_VERSATILE_RPM=60
		"""
		defaults = cls()  # hydrate the class-level defaults into an instance
		p = cls._ENV_PREFIX

		return cls(
			model_id=defaults.model_id,
			plan=defaults.plan,
			requests_per_minute=cls._env_int(
				f"{p}_REQUESTS_PER_MINUTE", defaults.requests_per_minute,
				),
			requests_per_day=cls._env_int(f"{p}_REQUESTS_PER_DAY", defaults.requests_per_day),
			tokens_per_minute=cls._env_int(f"{p}_TOKENS_PER_MINUTE", defaults.tokens_per_minute),
			tokens_per_day=cls._env_int(f"{p}_TOKENS_PER_DAY", defaults.tokens_per_day),
			audio_seconds_per_hour=cls._env_int(
				f"{p}_AUDIO_SECONDS_PER_HOUR", defaults.audio_seconds_per_hour,
				),
			audio_seconds_per_day=cls._env_int(
				f"{p}_AUDIO_SECONDS_PER_DAY", defaults.audio_seconds_per_day,
				),
			input_tokens_per_minute=cls._env_int(
				f"{p}_INPUT_TOKENS_PER_MINUTE", defaults.input_tokens_per_minute,
				),
			output_tokens_per_minute=cls._env_int(
				f"{p}_OUTPUT_TOKENS_PER_MINUTE", defaults.output_tokens_per_minute,
				),
			)
