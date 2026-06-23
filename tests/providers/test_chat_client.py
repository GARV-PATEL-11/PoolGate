"""
Tests for clients/chat_client.py.

The first test in this file (test_chat_client_is_instantiable) is the
canary / regression guard for the audit's most critical finding: ChatClient
used to be unable to instantiate at all because async_invoke/async_stream
were accidentally nested inside stream()'s body rather than defined at
class level, which left TextGenerationCapability's abstract methods
unimplemented. If this test ever fails again, treat it as the same class
of bug and check class-level method definitions with ast before anything
else.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.chat_client import ChatClient
from schemas.runtime import RequestConfig


@pytest.fixture
def chat_client() -> ChatClient:
    return ChatClient()


def test_chat_client_is_instantiable(chat_client):
    """Canary test for the critical bug found in the audit — must pass before anything else here can."""
    assert chat_client is not None
    assert hasattr(chat_client, "invoke")
    assert hasattr(chat_client, "async_invoke")
    assert hasattr(chat_client, "stream")
    assert hasattr(chat_client, "async_stream")


def _mock_completion(text: str, prompt_tokens=5, completion_tokens=3):
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=text), finish_reason="stop")]
    completion.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return completion


class TestInvoke:

    def test_invoke_returns_groq_response_on_success(self, chat_client, monkeypatch):
        """
        Regression test for a second bug found while fixing the first:
        invoke()'s `choice = _first_choice(completion, rid)` line was
        originally indented as part of the `except` block, so on the
        success path `choice` was never assigned and the function raised
        NameError on every successful call. Both bugs are fixed together.
        """
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion("hello")
        monkeypatch.setattr(chat_client, "_sync_sdk", lambda api_key: mock_sdk)

        response = chat_client.invoke(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
        )
        assert response.text == "hello"
        assert response.usage.prompt_tokens == 5

    def test_invoke_propagates_sdk_errors(self, chat_client, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.side_effect = RuntimeError("upstream failure")
        monkeypatch.setattr(chat_client, "_sync_sdk", lambda api_key: mock_sdk)
        monkeypatch.setattr(chat_client, "_handle_sdk_error", lambda exc, rid, key_id: None)

        with pytest.raises(RuntimeError):
            chat_client.invoke(
                api_key="gsk_test",
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "hi"}],
                config=RequestConfig(),
                session_id="s1",
                api_key_id="key_0",
            )


class TestAsyncInvoke:

    @pytest.mark.asyncio
    async def test_async_invoke_returns_groq_response(self, chat_client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_completion("async hello", 2, 4),
        )
        monkeypatch.setattr(chat_client, "_async_sdk", lambda api_key: mock_sdk)

        response = await chat_client.async_invoke(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
        )
        assert response.text == "async hello"
        assert response.usage.prompt_tokens == 2


class TestStreamUsesRealSdkShape:
    """
    Regression tests for a bug found while implementing the H5 streaming
    fix: the original code called client.chat.completions.stream(...) as a
    context manager, but the real groq SDK (verified against groq==1.4.0
    source) has no such method — only create(stream=True), which returns
    an iterable Stream object. These tests assert the client calls create()
    with stream=True, not a nonexistent .stream() method.
    """

    def test_stream_calls_create_with_stream_true(self, chat_client, monkeypatch):
        chunk1 = MagicMock(x_groq=None)
        chunk1.choices = [MagicMock(delta=MagicMock(content="Hel"))]
        chunk2 = MagicMock(x_groq=None)
        chunk2.choices = [MagicMock(delta=MagicMock(content="lo"))]

        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = iter([chunk1, chunk2])
        mock_stream.__exit__.return_value = False

        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = mock_stream
        monkeypatch.setattr(chat_client, "_sync_sdk", lambda api_key: mock_sdk)

        chunks = list(
            chat_client.stream(
                api_key="gsk_test",
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "hi"}],
                config=RequestConfig(),
                session_id="s1",
                api_key_id="key_0",
            ),
        )

        assert chunks == ["Hel", "lo"]
        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        # Confirms the bug is fixed: create() must be called, not a
        # nonexistent .stream() method.
        assert not hasattr(mock_sdk.chat.completions, "stream") or mock_sdk.chat.completions.create.called

    def test_stream_invokes_on_usage_callback_from_x_groq_final_chunk(
        self,
        chat_client,
        monkeypatch,
    ):
        """
        Regression test for the dropped-token-accounting bug: usage must be
        extracted from chunk.x_groq.usage on the final chunk (Groq's actual
        vendor extension, verified against SDK source — NOT the OpenAI-style
        stream_options mechanism, which this SDK version doesn't support).
        """
        final_chunk = MagicMock()
        final_chunk.choices = [MagicMock(delta=MagicMock(content=""))]
        final_chunk.x_groq = MagicMock()
        final_chunk.x_groq.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )

        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = iter([final_chunk])
        mock_stream.__exit__.return_value = False

        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = mock_stream
        monkeypatch.setattr(chat_client, "_sync_sdk", lambda api_key: mock_sdk)

        captured = {}
        list(
            chat_client.stream(
                api_key="gsk_test",
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "hi"}],
                config=RequestConfig(),
                session_id="s1",
                api_key_id="key_0",
                on_usage=lambda u: captured.setdefault("usage", u),
            ),
        )

        assert captured["usage"].prompt_tokens == 10
        assert captured["usage"].completion_tokens == 20


class TestAsyncStream:

    @pytest.mark.asyncio
    async def test_async_stream_yields_text_deltas(self, chat_client, monkeypatch):
        chunk1 = MagicMock(x_groq=None)
        chunk1.choices = [MagicMock(delta=MagicMock(content="He"))]
        chunk2 = MagicMock(x_groq=None)
        chunk2.choices = [MagicMock(delta=MagicMock(content="llo"))]

        async def _async_stream_cm():
            for c in [chunk1, chunk2]:
                yield c

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=_async_stream_cm())
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(return_value=mock_stream)
        monkeypatch.setattr(chat_client, "_async_sdk", lambda api_key: mock_sdk)

        chunks = []
        async for chunk in chat_client.async_stream(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
        ):
            chunks.append(chunk)

        assert chunks == ["He", "llo"]

    @pytest.mark.asyncio
    async def test_async_stream_calls_create_with_stream_true(self, chat_client, monkeypatch):
        async def _empty():
            return
            yield  # make it a generator

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=_empty())
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(return_value=mock_stream)
        monkeypatch.setattr(chat_client, "_async_sdk", lambda api_key: mock_sdk)

        async for _ in chat_client.async_stream(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
        ):
            pass

        call_kwargs = mock_sdk.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True

    @pytest.mark.asyncio
    async def test_async_invoke_propagates_sdk_errors(self, chat_client, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(side_effect=RuntimeError("async failure"))
        monkeypatch.setattr(chat_client, "_async_sdk", lambda api_key: mock_sdk)
        monkeypatch.setattr(chat_client, "_handle_sdk_error", lambda exc, rid, key_id: None)

        with pytest.raises(RuntimeError):
            await chat_client.async_invoke(
                api_key="gsk_test",
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "hi"}],
                config=RequestConfig(),
                session_id="s1",
                api_key_id="key_0",
            )


class TestStreamErrorPath:

    def test_stream_error_during_iteration_propagates(self, chat_client, monkeypatch):
        def _bad_iter():
            raise RuntimeError("stream failed mid-iteration")
            yield  # make it a generator

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=_bad_iter())
        mock_stream.__exit__ = MagicMock(return_value=False)

        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = mock_stream
        monkeypatch.setattr(chat_client, "_sync_sdk", lambda api_key: mock_sdk)
        monkeypatch.setattr(chat_client, "_handle_sdk_error", lambda exc, rid, key_id: None)

        with pytest.raises(RuntimeError):
            list(
                chat_client.stream(
                    api_key="gsk_test",
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": "hi"}],
                    config=RequestConfig(),
                    session_id="s1",
                    api_key_id="key_0",
                )
            )

    @pytest.mark.asyncio
    async def test_async_stream_invokes_on_usage_callback(self, chat_client, monkeypatch):
        final_chunk = MagicMock()
        final_chunk.choices = [MagicMock(delta=MagicMock(content=""))]
        final_chunk.x_groq = MagicMock()
        final_chunk.x_groq.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

        async def _stream_cm():
            yield final_chunk

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=_stream_cm())
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(return_value=mock_stream)
        monkeypatch.setattr(chat_client, "_async_sdk", lambda api_key: mock_sdk)

        captured = {}
        async for _ in chat_client.async_stream(
            api_key="gsk_test",
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            config=RequestConfig(),
            session_id="s1",
            api_key_id="key_0",
            on_usage=lambda u: captured.setdefault("usage", u),
        ):
            pass

        assert captured["usage"].prompt_tokens == 10
        assert captured["usage"].completion_tokens == 20

    @pytest.mark.asyncio
    async def test_async_stream_error_propagates(self, chat_client, monkeypatch):
        async def _bad_gen():
            raise RuntimeError("async stream failed")
            yield  # make it an async generator

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=_bad_gen())
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(return_value=mock_stream)
        monkeypatch.setattr(chat_client, "_async_sdk", lambda api_key: mock_sdk)
        monkeypatch.setattr(chat_client, "_handle_sdk_error", lambda exc, rid, key_id: None)

        with pytest.raises(RuntimeError):
            async for _ in chat_client.async_stream(
                api_key="gsk_test",
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "hi"}],
                config=RequestConfig(),
                session_id="s1",
                api_key_id="key_0",
            ):
                pass
