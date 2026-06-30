"""Groq-specific capability ABCs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from typing import Any


class TextGenerationCapability(ABC):
    @abstractmethod
    def invoke(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    async def async_invoke(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def stream(self, *args: Any, **kwargs: Any) -> Generator[str, None, None]: ...
    @abstractmethod
    def async_stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[str, None]: ...


class StructuredGenerationCapability(ABC):
    @abstractmethod
    def invoke_structured(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    async def async_invoke_structured(self, *args: Any, **kwargs: Any) -> Any: ...


class ToolCallingCapability(ABC):
    @abstractmethod
    def invoke_tools(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    async def async_invoke_tools(self, *args: Any, **kwargs: Any) -> Any: ...


class ModerationCapability(ABC):
    @abstractmethod
    def moderate(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    async def async_moderate(self, *args: Any, **kwargs: Any) -> Any: ...


class TranscriptionCapability(ABC):
    @abstractmethod
    def transcribe(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    async def async_transcribe(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def translate(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    async def async_translate(self, *args: Any, **kwargs: Any) -> Any: ...


class SynthesisCapability(ABC):
    @abstractmethod
    def synthesize(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    async def async_synthesize(self, *args: Any, **kwargs: Any) -> Any: ...
