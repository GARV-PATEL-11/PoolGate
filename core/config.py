"""
Configuration loader.
All settings are pulled from environment variables so no secrets live in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from exceptions.configuration import ConfigurationError, EnvironmentParseError


@dataclass
class GroqConfig:
	"""Central configuration object. Populated once at import time."""

	# Raw API keys — loaded from GROQ_API_KEY_01 … GROQ_API_KEY_N
	api_keys: list[str] = field(default_factory=list)

	# Key health thresholds
	max_rpm_per_key: int = 30  # requests per minute before penalty
	max_active_requests: int = 10  # concurrent requests per key
	cooldown_seconds: float = 60.0  # how long to cool down after 429
	failure_threshold: int = 5  # consecutive failures before FAILED status
	latency_penalty_threshold: float = 3.0  # seconds

	# Scheduler / pool
	default_concurrency: int = 20  # batch semaphore default
	health_check_interval: float = 30.0  # seconds between background checks

	# Retry defaults
	max_retries: int = 3
	base_backoff: float = 1.0
	max_backoff: float = 30.0

	# Session
	session_ttl_hours: int = 24

	# Observability
	debug_mode: bool = False
	log_level: str = "INFO"

	@classmethod
	def from_env(cls) -> GroqConfig:
		def env_int(name: str, default: str) -> int:
			raw_value = os.environ.get(name, default)
			try:
				return int(raw_value)
			except ValueError as exc:
				raise EnvironmentParseError(
					f"{name} must be an integer; got {raw_value!r}.",
					var_name=name,
					raw_value=raw_value,
					expected=int,
					) from exc

		def env_float(name: str, default: str) -> float:
			raw_value = os.environ.get(name, default)
			try:
				return float(raw_value)
			except ValueError as exc:
				raise EnvironmentParseError(
					f"{name} must be a float; got {raw_value!r}.",
					var_name=name,
					raw_value=raw_value,
					expected=float,
					) from exc

		def env_bool(name: str, default: str) -> bool:
			raw_value = os.environ.get(name, default)
			return raw_value.strip().lower() in ("1", "true", "yes")

		# --- Key loading ---------------------------------------------------
		total_keys = env_int("TOTAL_GROQ_KEYS", "1")
		keys: list[str] = []
		for i in range(1, total_keys + 1):
			var_name = f"GROQ_API_KEY_{i:02d}"
			value = os.environ.get(var_name, "").strip()
			if value:
				keys.append(value)

		if not keys:
			raise ConfigurationError(
				f"No valid Groq API keys found. "
				f"Set TOTAL_GROQ_KEYS and populate GROQ_API_KEY_01 … "
				f"GROQ_API_KEY_{total_keys:02d} in your environment.",
				)
		# -------------------------------------------------------------------

		return cls(
			api_keys=keys,
			max_rpm_per_key=env_int("GROQ_MAX_RPM", "30"),
			max_active_requests=env_int("GROQ_MAX_ACTIVE", "10"),
			cooldown_seconds=env_float("GROQ_COOLDOWN_SECS", "60"),
			failure_threshold=env_int("GROQ_FAILURE_THRESHOLD", "5"),
			default_concurrency=env_int("GROQ_BATCH_CONCURRENCY", "20"),
			max_retries=env_int("GROQ_MAX_RETRIES", "3"),
			base_backoff=env_float("GROQ_BASE_BACKOFF", "1.0"),
			max_backoff=env_float("GROQ_MAX_BACKOFF", "30.0"),
			session_ttl_hours=env_int("GROQ_SESSION_TTL_HOURS", "24"),
			debug_mode=env_bool("GROQ_DEBUG", "false"),
			log_level=os.environ.get("GROQ_LOG_LEVEL", "INFO").upper(),
			)
