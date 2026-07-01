"""GeminiConfig — configuration loader for Google Gemini key pool."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from poolgate.core.paths import PathConfig
from poolgate.exceptions.configuration import ConfigurationError, EnvironmentParseError


@dataclass
class GeminiConfig:
    """Configuration for the Gemini key pool. Mirrors GroqConfig field layout."""

    api_keys: list[str] = field(default_factory=list)

    max_rpm_per_key: int = 5
    max_active_requests: int = 5
    cooldown_seconds: float = 60.0
    failure_threshold: int = 3
    latency_penalty_threshold: float = 3.0

    default_concurrency: int = 10
    health_check_interval: float = 30.0

    max_retries: int = 3
    base_backoff: float = 1.0
    max_backoff: float = 30.0

    session_ttl_hours: int = 24

    debug_mode: bool = False
    log_level: str = "INFO"

    paths: PathConfig = field(default_factory=lambda: PathConfig(namespace="gemini"))

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        def env_int(name: str, default: str) -> int:
            raw = os.environ.get(name, default)
            try:
                return int(raw)
            except ValueError as exc:
                raise EnvironmentParseError(
                    f"{name} must be an integer; got {raw!r}.",
                    var_name=name,
                    raw_value=raw,
                    expected=int,
                ) from exc

        def env_float(name: str, default: str) -> float:
            raw = os.environ.get(name, default)
            try:
                return float(raw)
            except ValueError as exc:
                raise EnvironmentParseError(
                    f"{name} must be a float; got {raw!r}.",
                    var_name=name,
                    raw_value=raw,
                    expected=float,
                ) from exc

        def env_bool(name: str, default: str) -> bool:
            return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")

        total_keys = env_int("TOTAL_GEMINI_KEYS", "1")
        keys: list[str] = []
        for i in range(1, total_keys + 1):
            var_name = f"GEMINI_API_KEY_{i:02d}"
            value = os.environ.get(var_name, "").strip()
            if value:
                keys.append(value)

        if not keys:
            raise ConfigurationError(
                f"No valid Gemini API keys found. "
                f"Set TOTAL_GEMINI_KEYS and GEMINI_API_KEY_01 … GEMINI_API_KEY_{total_keys:02d}.",
            )

        return cls(
            api_keys=keys,
            max_rpm_per_key=env_int("GEMINI_MAX_RPM", "5"),
            max_active_requests=env_int("GEMINI_MAX_ACTIVE", "5"),
            cooldown_seconds=env_float("GEMINI_COOLDOWN_SECS", "60"),
            failure_threshold=env_int("GEMINI_FAILURE_THRESHOLD", "3"),
            default_concurrency=env_int("GEMINI_BATCH_CONCURRENCY", "10"),
            max_retries=env_int("GEMINI_MAX_RETRIES", "3"),
            base_backoff=env_float("GEMINI_BASE_BACKOFF", "1.0"),
            max_backoff=env_float("GEMINI_MAX_BACKOFF", "30.0"),
            session_ttl_hours=env_int("GEMINI_SESSION_TTL_HOURS", "24"),
            debug_mode=env_bool("GEMINI_DEBUG", "false"),
            log_level=os.environ.get("GEMINI_LOG_LEVEL", "INFO").upper(),
        )
