"""Unit tests for poolgate/providers/groq/capabilities.py — ABC interface contracts."""

from __future__ import annotations

import pytest

# Concrete capability classes (were *Client in old codebase)
from poolgate.capabilities import (
    ChatCapability as ChatClient,
    ModerationCapability as ModerationClient,
    StructuredCapability as StructuredClient,
    SynthesisCapability as SynthesisClient,
    ToolCapability as ToolClient,
    TranscriptionCapability as TranscriptionClient,
)

# Abstract base classes (ABCs) from providers layer
from poolgate.providers.groq.capabilities import (
    ModerationCapability,
    StructuredGenerationCapability,
    SynthesisCapability,
    TextGenerationCapability,
    ToolCallingCapability,
    TranscriptionCapability,
)

# ---------------------------------------------------------------------------
# ABCs cannot be instantiated
# ---------------------------------------------------------------------------


class TestAbstractBaseClasses:

    def test_text_generation_capability_is_abstract(self):
        with pytest.raises(TypeError):
            TextGenerationCapability()  # type: ignore[abstract]

    def test_structured_generation_capability_is_abstract(self):
        with pytest.raises(TypeError):
            StructuredGenerationCapability()  # type: ignore[abstract]

    def test_tool_calling_capability_is_abstract(self):
        with pytest.raises(TypeError):
            ToolCallingCapability()  # type: ignore[abstract]

    def test_moderation_capability_is_abstract(self):
        with pytest.raises(TypeError):
            ModerationCapability()  # type: ignore[abstract]

    def test_transcription_capability_is_abstract(self):
        with pytest.raises(TypeError):
            TranscriptionCapability()  # type: ignore[abstract]

    def test_synthesis_capability_is_abstract(self):
        with pytest.raises(TypeError):
            SynthesisCapability()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Concrete clients satisfy their capability ABCs
# ---------------------------------------------------------------------------


class TestConcreteClientInheritance:

    def test_chat_client_is_text_generation_capability(self):
        assert issubclass(ChatClient, TextGenerationCapability)
        assert isinstance(ChatClient(), TextGenerationCapability)

    def test_structured_client_is_structured_generation_capability(self):
        assert issubclass(StructuredClient, StructuredGenerationCapability)
        assert isinstance(StructuredClient(), StructuredGenerationCapability)

    def test_tool_client_is_tool_calling_capability(self):
        assert issubclass(ToolClient, ToolCallingCapability)
        assert isinstance(ToolClient(), ToolCallingCapability)

    def test_moderation_client_is_moderation_capability(self):
        assert issubclass(ModerationClient, ModerationCapability)
        assert isinstance(ModerationClient(), ModerationCapability)

    def test_transcription_client_is_transcription_capability(self):
        assert issubclass(TranscriptionClient, TranscriptionCapability)
        assert isinstance(TranscriptionClient(), TranscriptionCapability)

    def test_synthesis_client_is_synthesis_capability(self):
        assert issubclass(SynthesisClient, SynthesisCapability)
        assert isinstance(SynthesisClient(), SynthesisCapability)


# ---------------------------------------------------------------------------
# Method names declared on ABCs
# ---------------------------------------------------------------------------


class TestAbstractMethodNames:

    def test_text_generation_has_invoke(self):
        assert hasattr(TextGenerationCapability, "invoke")

    def test_text_generation_has_async_invoke(self):
        assert hasattr(TextGenerationCapability, "async_invoke")

    def test_text_generation_has_stream(self):
        assert hasattr(TextGenerationCapability, "stream")

    def test_text_generation_has_async_stream(self):
        assert hasattr(TextGenerationCapability, "async_stream")

    def test_structured_generation_has_invoke_structured(self):
        assert hasattr(StructuredGenerationCapability, "invoke_structured")

    def test_structured_generation_has_async_invoke_structured(self):
        assert hasattr(StructuredGenerationCapability, "async_invoke_structured")

    def test_tool_calling_has_invoke_tools(self):
        assert hasattr(ToolCallingCapability, "invoke_tools")

    def test_tool_calling_has_async_invoke_tools(self):
        assert hasattr(ToolCallingCapability, "async_invoke_tools")

    def test_moderation_has_moderate(self):
        assert hasattr(ModerationCapability, "moderate")

    def test_moderation_has_async_moderate(self):
        assert hasattr(ModerationCapability, "async_moderate")

    def test_transcription_has_transcribe(self):
        assert hasattr(TranscriptionCapability, "transcribe")

    def test_transcription_has_async_transcribe(self):
        assert hasattr(TranscriptionCapability, "async_transcribe")

    def test_transcription_has_translate(self):
        assert hasattr(TranscriptionCapability, "translate")

    def test_transcription_has_async_translate(self):
        assert hasattr(TranscriptionCapability, "async_translate")

    def test_synthesis_has_synthesize(self):
        assert hasattr(SynthesisCapability, "synthesize")

    def test_synthesis_has_async_synthesize(self):
        assert hasattr(SynthesisCapability, "async_synthesize")


# ---------------------------------------------------------------------------
# Concrete clients expose all required methods
# ---------------------------------------------------------------------------


class TestConcreteClientMethods:

    def test_chat_client_has_all_text_generation_methods(self):
        for method in ("invoke", "async_invoke", "stream", "async_stream"):
            assert callable(getattr(ChatClient, method)), f"ChatClient.{method} missing"

    def test_structured_client_has_all_structured_methods(self):
        for method in ("invoke_structured", "async_invoke_structured"):
            assert callable(getattr(StructuredClient, method)), f"StructuredClient.{method} missing"

    def test_tool_client_has_all_tool_methods(self):
        for method in ("invoke_tools", "async_invoke_tools"):
            assert callable(getattr(ToolClient, method)), f"ToolClient.{method} missing"

    def test_moderation_client_has_all_moderation_methods(self):
        for method in ("moderate", "async_moderate"):
            assert callable(getattr(ModerationClient, method)), f"ModerationClient.{method} missing"

    def test_transcription_client_has_all_transcription_methods(self):
        for method in (
            "transcribe",
            "async_transcribe",
            "translate",
            "async_translate",
        ):
            assert callable(getattr(TranscriptionClient, method)), f"TranscriptionClient.{method} missing"

    def test_synthesis_client_has_all_synthesis_methods(self):
        for method in ("synthesize", "async_synthesize"):
            assert callable(getattr(SynthesisClient, method)), f"SynthesisClient.{method} missing"


# ---------------------------------------------------------------------------
# Abstract method stub bodies are reachable via super()
# ---------------------------------------------------------------------------


class TestAbstractMethodBodies:
    """
    The `...` body of each abstract method is an executable statement covered
    only when a concrete subclass calls super().  These tests verify that
    calling the stubs via super() returns None (the implicit return when the
    body is `...`) and does not raise, documenting the stub contract.
    """

    def test_text_generation_invoke_stub_returns_none(self):
        class _Stub(TextGenerationCapability):
            def invoke(self, *a, **kw):
                return super().invoke(*a, **kw)

            async def async_invoke(self, *a, **kw):
                return await super().async_invoke(*a, **kw)

            def stream(self, *a, **kw):
                return super().stream(*a, **kw)

            async def async_stream(self, *a, **kw):
                return await super().async_stream(*a, **kw)

        stub = _Stub()
        assert stub.invoke() is None

    def test_text_generation_stream_stub_returns_none(self):
        class _Stub(TextGenerationCapability):
            def invoke(self, *a, **kw):
                return super().invoke(*a, **kw)

            async def async_invoke(self, *a, **kw):
                return await super().async_invoke(*a, **kw)

            def stream(self, *a, **kw):
                return super().stream(*a, **kw)

            async def async_stream(self, *a, **kw):
                return await super().async_stream(*a, **kw)

        stub = _Stub()
        assert stub.stream() is None

    @pytest.mark.asyncio
    async def test_text_generation_async_invoke_stub_returns_none(self):
        class _Stub(TextGenerationCapability):
            def invoke(self, *a, **kw):
                return super().invoke(*a, **kw)

            async def async_invoke(self, *a, **kw):
                return await super().async_invoke(*a, **kw)

            def stream(self, *a, **kw):
                return super().stream(*a, **kw)

            async def async_stream(self, *a, **kw):
                return await super().async_stream(*a, **kw)

        stub = _Stub()
        assert await stub.async_invoke() is None

    def test_text_generation_async_stream_stub_returns_none(self):
        class _Stub(TextGenerationCapability):
            def invoke(self, *a, **kw):
                return super().invoke(*a, **kw)

            async def async_invoke(self, *a, **kw):
                return await super().async_invoke(*a, **kw)

            def stream(self, *a, **kw):
                return super().stream(*a, **kw)

            def async_stream(self, *a, **kw):
                return super().async_stream(*a, **kw)

        stub = _Stub()
        assert stub.async_stream() is None

    def test_structured_generation_stubs_return_none(self):
        class _Stub(StructuredGenerationCapability):
            def invoke_structured(self, *a, **kw):
                return super().invoke_structured(*a, **kw)

            async def async_invoke_structured(self, *a, **kw):
                return await super().async_invoke_structured(*a, **kw)

        stub = _Stub()
        assert stub.invoke_structured() is None

    @pytest.mark.asyncio
    async def test_structured_generation_async_stub_returns_none(self):
        class _Stub(StructuredGenerationCapability):
            def invoke_structured(self, *a, **kw):
                return super().invoke_structured(*a, **kw)

            async def async_invoke_structured(self, *a, **kw):
                return await super().async_invoke_structured(*a, **kw)

        stub = _Stub()
        assert await stub.async_invoke_structured() is None

    def test_tool_calling_stubs_return_none(self):
        class _Stub(ToolCallingCapability):
            def invoke_tools(self, *a, **kw):
                return super().invoke_tools(*a, **kw)

            async def async_invoke_tools(self, *a, **kw):
                return await super().async_invoke_tools(*a, **kw)

        stub = _Stub()
        assert stub.invoke_tools() is None

    @pytest.mark.asyncio
    async def test_tool_calling_async_stub_returns_none(self):
        class _Stub(ToolCallingCapability):
            def invoke_tools(self, *a, **kw):
                return super().invoke_tools(*a, **kw)

            async def async_invoke_tools(self, *a, **kw):
                return await super().async_invoke_tools(*a, **kw)

        stub = _Stub()
        assert await stub.async_invoke_tools() is None

    def test_moderation_stubs_return_none(self):
        class _Stub(ModerationCapability):
            def moderate(self, *a, **kw):
                return super().moderate(*a, **kw)

            async def async_moderate(self, *a, **kw):
                return await super().async_moderate(*a, **kw)

        stub = _Stub()
        assert stub.moderate() is None

    @pytest.mark.asyncio
    async def test_moderation_async_stub_returns_none(self):
        class _Stub(ModerationCapability):
            def moderate(self, *a, **kw):
                return super().moderate(*a, **kw)

            async def async_moderate(self, *a, **kw):
                return await super().async_moderate(*a, **kw)

        stub = _Stub()
        assert await stub.async_moderate() is None

    def test_transcription_stubs_return_none(self):
        class _Stub(TranscriptionCapability):
            def transcribe(self, *a, **kw):
                return super().transcribe(*a, **kw)

            async def async_transcribe(self, *a, **kw):
                return await super().async_transcribe(*a, **kw)

            def translate(self, *a, **kw):
                return super().translate(*a, **kw)

            async def async_translate(self, *a, **kw):
                return await super().async_translate(*a, **kw)

        stub = _Stub()
        assert stub.transcribe() is None
        assert stub.translate() is None

    @pytest.mark.asyncio
    async def test_transcription_async_stubs_return_none(self):
        class _Stub(TranscriptionCapability):
            def transcribe(self, *a, **kw):
                return super().transcribe(*a, **kw)

            async def async_transcribe(self, *a, **kw):
                return await super().async_transcribe(*a, **kw)

            def translate(self, *a, **kw):
                return super().translate(*a, **kw)

            async def async_translate(self, *a, **kw):
                return await super().async_translate(*a, **kw)

        stub = _Stub()
        assert await stub.async_transcribe() is None
        assert await stub.async_translate() is None

    def test_synthesis_stubs_return_none(self):
        class _Stub(SynthesisCapability):
            def synthesize(self, *a, **kw):
                return super().synthesize(*a, **kw)

            async def async_synthesize(self, *a, **kw):
                return await super().async_synthesize(*a, **kw)

        stub = _Stub()
        assert stub.synthesize() is None

    @pytest.mark.asyncio
    async def test_synthesis_async_stub_returns_none(self):
        class _Stub(SynthesisCapability):
            def synthesize(self, *a, **kw):
                return super().synthesize(*a, **kw)

            async def async_synthesize(self, *a, **kw):
                return await super().async_synthesize(*a, **kw)

        stub = _Stub()
        assert await stub.async_synthesize() is None
