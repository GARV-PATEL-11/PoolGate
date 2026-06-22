"""
capabilities.py — Abstract base classes (capability interfaces) for PoolGate clients.

Each ABC defines the exact public method contract for a single model capability.
Concrete clients inherit one or more of these interfaces.

Capability               Methods                                          Supported by
────────────────────────────────────────────────────────────────────────────────────────
TextGenerationCapability  invoke / async_invoke / stream / async_stream            chat LLMs, compound
StructuredGenCapability   invoke_structured / async_invoke_structured         chat LLMs (JSON mode)
ToolCallingCapability     invoke_tools / async_invoke_tools                   chat LLMs, compound
ModerationCapability      moderate / async_moderate                           prompt-guard, safeguard
TranscriptionCapability   transcribe / async_transcribe / translate /         whisper-*
                          async_translate
SynthesisCapability       synthesize / async_synthesize                       orpheus-*
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from typing import Any


class TextGenerationCapability(ABC):
	"""
	Chat / instruction-following text generation with optional streaming.

	Implemented by: ChatClient
	"""

	@abstractmethod
	def invoke(self, *args: Any, **kwargs: Any) -> Any:
		"""Blocking single-turn completion. Returns GroqResponse."""
		...

	@abstractmethod
	async def async_invoke(self, *args: Any, **kwargs: Any) -> Any:
		"""Async single-turn completion. Returns GroqResponse."""
		...

	@abstractmethod
	def stream(self, *args: Any, **kwargs: Any) -> Generator[str, None, None]:
		"""Blocking token stream. Yields str chunks."""
		...

	@abstractmethod
	async def async_stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
		"""Async token stream. Yields str chunks."""
		...


class StructuredGenerationCapability(ABC):
	"""
	JSON-mode / schema-constrained structured output.

	Implemented by: StructuredClient
	"""

	@abstractmethod
	def invoke_structured(self, *args: Any, **kwargs: Any) -> Any:
		"""Blocking structured completion. Returns GroqResponse (text is valid JSON)."""
		...

	@abstractmethod
	async def async_invoke_structured(self, *args: Any, **kwargs: Any) -> Any:
		"""Async structured completion. Returns GroqResponse (text is valid JSON)."""
		...


class ToolCallingCapability(ABC):
	"""
	Function / tool calling with a configurable tool_choice policy.

	Implemented by: ToolClient
	"""

	@abstractmethod
	def invoke_tools(self, *args: Any, **kwargs: Any) -> Any:
		"""Blocking tool-calling invocation. Returns GroqResponse."""
		...

	@abstractmethod
	async def async_invoke_tools(self, *args: Any, **kwargs: Any) -> Any:
		"""Async tool-calling invocation. Returns GroqResponse."""
		...


class ModerationCapability(ABC):
	"""
	Safety classification — returns structured labels, not freeform text.

	Implemented by: ModerationClient
	"""

	@abstractmethod
	def moderate(self, *args: Any, **kwargs: Any) -> Any:
		"""Blocking content classification. Returns ModerationResult."""
		...

	@abstractmethod
	async def async_moderate(self, *args: Any, **kwargs: Any) -> Any:
		"""Async content classification. Returns ModerationResult."""
		...


class TranscriptionCapability(ABC):
	"""
	Speech-to-text transcription and optional English translation.

	Implemented by: TranscriptionClient
	"""

	@abstractmethod
	def transcribe(self, *args: Any, **kwargs: Any) -> Any:
		"""Blocking transcription in the source language. Returns TranscriptionResult."""
		...

	@abstractmethod
	async def async_transcribe(self, *args: Any, **kwargs: Any) -> Any:
		"""Async transcription in the source language. Returns TranscriptionResult."""
		...

	@abstractmethod
	def translate(self, *args: Any, **kwargs: Any) -> Any:
		"""Blocking transcription translated to English. Returns TranscriptionResult."""
		...

	@abstractmethod
	async def async_translate(self, *args: Any, **kwargs: Any) -> Any:
		"""Async transcription translated to English. Returns TranscriptionResult."""
		...


class SynthesisCapability(ABC):
	"""
	Text-to-speech audio synthesis.

	Implemented by: SynthesisClient
	"""

	@abstractmethod
	def synthesize(self, *args: Any, **kwargs: Any) -> Any:
		"""Blocking TTS synthesis. Returns SynthesisResult (audio bytes)."""
		...

	@abstractmethod
	async def async_synthesize(self, *args: Any, **kwargs: Any) -> Any:
		"""Async TTS synthesis. Returns SynthesisResult (audio bytes)."""
		...
