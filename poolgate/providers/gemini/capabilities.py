"""Gemini-specific capability ABCs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from typing import Any


class GeminiTextGenerationCapability(ABC):
    @abstractmethod
    def invoke(self, *args: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    async def async_invoke(self, *args: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def stream(self, *args: Any, **kwargs: Any) -> Generator[str, None, None]: ...

    @abstractmethod
    def async_stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[str, None]: ...


class GeminiStructuredGenerationCapability(ABC):
    @abstractmethod
    def invoke_structured(self, *args: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    async def async_invoke_structured(self, *args: Any, **kwargs: Any) -> Any: ...


class GeminiToolCallingCapability(ABC):
    @abstractmethod
    def invoke_tools(self, *args: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    async def async_invoke_tools(self, *args: Any, **kwargs: Any) -> Any: ...
