"""PoolGate — multi-key API pool for Groq and Google Gemini."""

from typing import Any

from poolgate.core.config import GroqConfig
from poolgate.core.gemini_config import GeminiConfig
from poolgate.core.paths import PathConfig


def __getattr__(name: str) -> Any:
    if name == "GroqService":
        from poolgate.services.provider import GroqService

        return GroqService
    if name == "GeminiService":
        from poolgate.services.gemini_provider import GeminiService

        return GeminiService
    raise AttributeError(f"module 'poolgate' has no attribute {name!r}")


__all__ = ["GeminiConfig", "GeminiService", "GroqConfig", "GroqService", "PathConfig"]
