"""PoolGate — multi-key Groq API pool."""

from typing import Any

from poolgate.core.config import GroqConfig
from poolgate.core.paths import PathConfig


def __getattr__(name: str) -> Any:
    if name == "GroqService":
        from poolgate.services.provider import GroqService

        return GroqService
    raise AttributeError(f"module 'poolgate' has no attribute {name!r}")


__all__ = ["GroqService", "GroqConfig", "PathConfig"]
