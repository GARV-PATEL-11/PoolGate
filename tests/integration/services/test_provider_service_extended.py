"""
Extended integration tests for services/provider_service.py.

Covers async methods (async_invoke, async_chat, async_structured, async_stream),
capability methods (moderate, transcribe, translate, synthesize, invoke_tools and
their async variants), session helpers, and error paths not exercised by the base
integration test suite.

All Groq SDK calls are mocked — no real network calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from poolgate.capabilities import (
    ModerationResult,
    SynthesisResult,
    TranscriptionResult,
)
from poolgate.core.config import GroqConfig
from poolgate.core.paths import PathConfig
from poolgate.exceptions import EmptyKeyPoolError
from poolgate.exceptions.keys import APIKeyDisabledError, NoAvailableAPIKeyError
from poolgate.exceptions.output import SessionExpiredError
from poolgate.exceptions.output import StructuredOutputError
from poolgate.exceptions.request import MissingPromptError
from poolgate.exceptions.response import RetryExhaustedError
from poolgate.schemas.common.runtime import RequestConfig, TokenUsage
from poolgate.services.provider import GroqService

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setenv("TOTAL_GROQ_KEYS", "2")
    monkeypatch.setenv("GROQ_API_KEY_01", "gsk_key_1")
    monkeypatch.setenv("GROQ_API_KEY_02", "gsk_key_2")
    return GroqService()


def _mock_completion(text: str, prompt_tokens: int = 5, completion_tokens: int = 3):
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=text), finish_reason="stop")]
    completion.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return completion


def _mock_mod_result(label: str = "SAFE") -> ModerationResult:
    usage = TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
    return ModerationResult(
        label=label,
        raw_text=label,
        model="meta-llama/llama-prompt-guard-2-86m",
        usage=usage,
        latency=0.05,
        session_id="s1",
        request_id="r1",
        api_key_id="key_0",
    )


def _mock_transcription_result() -> TranscriptionResult:
    return TranscriptionResult(
        text="hello world",
        model="whisper-large-v3",
        latency=0.1,
        session_id="s1",
        request_id="r1",
        api_key_id="key_0",
    )


def _mock_synthesis_result() -> SynthesisResult:
    return SynthesisResult(
        audio=b"\x00\x01\x02\x03",
        model="canopylabs/orpheus-v1-english",
        voice="Aria",
        response_format="mp3",
        latency=0.2,
        session_id="s1",
        request_id="r1",
        api_key_id="key_0",
    )


# ---------------------------------------------------------------------------
# async_invoke
# ---------------------------------------------------------------------------


class TestAsyncInvoke:

    @pytest.mark.asyncio
    async def test_async_invoke_returns_response(self, service, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_completion("async answer"),
        )
        monkeypatch.setattr(service._chat_client, "_async_sdk", lambda api_key: mock_sdk)

        response = await service.async_invoke("Hello", model="llama-3.3-70b-versatile")
        assert response.text == "async answer"

    @pytest.mark.asyncio
    async def test_async_invoke_with_system_prompt(self, service, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_completion("sys response"),
        )
        monkeypatch.setattr(service._chat_client, "_async_sdk", lambda api_key: mock_sdk)

        response = await service.async_invoke(
            "Hello",
            model="llama-3.3-70b-versatile",
            system="You are helpful.",
        )
        assert response.text == "sys response"
        messages = mock_sdk.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"


# ---------------------------------------------------------------------------
# async_chat
# ---------------------------------------------------------------------------


class TestAsyncChat:

    @pytest.mark.asyncio
    async def test_async_chat_returns_response(self, service, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_completion("async chat"),
        )
        monkeypatch.setattr(service._chat_client, "_async_sdk", lambda api_key: mock_sdk)

        messages = [{"role": "user", "content": "hey"}]
        response = await service.async_chat(messages, model="llama-3.3-70b-versatile")
        assert response.text == "async chat"

    @pytest.mark.asyncio
    async def test_async_chat_updates_global_stats(self, service, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_completion("ok"),
        )
        monkeypatch.setattr(service._chat_client, "_async_sdk", lambda api_key: mock_sdk)

        before = service.get_global_stats()["total_requests"]
        await service.async_chat(
            [{"role": "user", "content": "Q"}],
            model="llama-3.3-70b-versatile",
        )
        assert service.get_global_stats()["total_requests"] == before + 1


# ---------------------------------------------------------------------------
# async_structured
# ---------------------------------------------------------------------------


class TestAsyncStructured:

    @pytest.mark.asyncio
    async def test_async_structured_returns_pydantic_model(self, service, monkeypatch):
        class Answer(BaseModel):
            value: int

        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_completion('{"value": 99}'),
        )
        monkeypatch.setattr(service._structured_client, "_async_sdk", lambda api_key: mock_sdk)

        result = await service.async_structured(
            "What is 9*11?",
            Answer,
            model="llama-3.3-70b-versatile",
        )
        assert isinstance(result, Answer)
        assert result.value == 99

    @pytest.mark.asyncio
    async def test_async_structured_raises_on_bad_json(self, service, monkeypatch):
        class Schema(BaseModel):
            x: int

        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_completion("not json"),
        )
        monkeypatch.setattr(service._structured_client, "_async_sdk", lambda api_key: mock_sdk)

        with pytest.raises(StructuredOutputError):
            await service.async_structured("prompt", Schema, model="llama-3.3-70b-versatile")

    @pytest.mark.asyncio
    async def test_async_structured_strips_code_fence(self, service, monkeypatch):
        class Answer(BaseModel):
            result: str

        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_completion('```json\n{"result": "ok"}\n```'),
        )
        monkeypatch.setattr(service._structured_client, "_async_sdk", lambda api_key: mock_sdk)

        result = await service.async_structured(
            "prompt",
            Answer,
            model="llama-3.3-70b-versatile",
        )
        assert result.result == "ok"


# ---------------------------------------------------------------------------
# stream with messages argument
# ---------------------------------------------------------------------------


class TestStreamMessages:

    def test_stream_with_messages_yields_chunks(self, service, monkeypatch):
        chunk1 = MagicMock(x_groq=None)
        chunk1.choices = [MagicMock(delta=MagicMock(content="chunk"))]

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter([chunk1]))
        mock_stream.__exit__ = MagicMock(return_value=False)

        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = mock_stream
        monkeypatch.setattr(service._chat_client, "_sync_sdk", lambda api_key: mock_sdk)

        chunks = list(
            service.stream(
                messages=[{"role": "user", "content": "hi"}],
                model="llama-3.3-70b-versatile",
            )
        )
        assert "chunk" in chunks

    def test_stream_raises_when_neither_prompt_nor_messages(self, service):
        with pytest.raises(MissingPromptError):
            list(service.stream(model="llama-3.3-70b-versatile"))


# ---------------------------------------------------------------------------
# async_stream
# ---------------------------------------------------------------------------


class TestAsyncStream:

    @pytest.mark.asyncio
    async def test_async_stream_yields_chunks(self, service, monkeypatch):
        async def _gen():
            chunk = MagicMock(x_groq=None)
            chunk.choices = [MagicMock(delta=MagicMock(content="async chunk"))]
            yield chunk

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=_gen())
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(return_value=mock_stream)
        monkeypatch.setattr(service._chat_client, "_async_sdk", lambda api_key: mock_sdk)

        chunks = []
        async for chunk in service.async_stream("Say hi", model="llama-3.3-70b-versatile"):
            chunks.append(chunk)
        assert "async chunk" in chunks

    @pytest.mark.asyncio
    async def test_async_stream_raises_missing_prompt(self, service):
        with pytest.raises(MissingPromptError):
            async for _ in service.async_stream(model="llama-3.3-70b-versatile"):
                pass

    @pytest.mark.asyncio
    async def test_async_stream_with_messages_arg(self, service, monkeypatch):
        async def _gen():
            chunk = MagicMock(x_groq=None)
            chunk.choices = [MagicMock(delta=MagicMock(content="msg chunk"))]
            yield chunk

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=_gen())
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(return_value=mock_stream)
        monkeypatch.setattr(service._chat_client, "_async_sdk", lambda api_key: mock_sdk)

        chunks = []
        async for chunk in service.async_stream(
            messages=[{"role": "user", "content": "Q"}],
            model="llama-3.3-70b-versatile",
        ):
            chunks.append(chunk)
        assert "msg chunk" in chunks


# ---------------------------------------------------------------------------
# invoke_tools / async_invoke_tools
# ---------------------------------------------------------------------------


class TestInvokeTools:

    def test_invoke_tools_returns_response(self, service, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion("tool response")
        monkeypatch.setattr(service._tool_client, "_sync_sdk", lambda api_key: mock_sdk)

        tools = [{"type": "function", "function": {"name": "get_time", "parameters": {}}}]
        messages = [{"role": "user", "content": "What time is it?"}]
        response = service.invoke_tools(
            messages,
            tools,
            model="llama-3.3-70b-versatile",
        )
        assert response.text == "tool response"

    @pytest.mark.asyncio
    async def test_async_invoke_tools_returns_response(self, service, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            return_value=_mock_completion("async tool response"),
        )
        monkeypatch.setattr(service._tool_client, "_async_sdk", lambda api_key: mock_sdk)

        tools = [{"type": "function", "function": {"name": "fn", "parameters": {}}}]
        messages = [{"role": "user", "content": "call fn"}]
        response = await service.async_invoke_tools(
            messages,
            tools,
            model="llama-3.3-70b-versatile",
        )
        assert response.text == "async tool response"


# ---------------------------------------------------------------------------
# moderate / async_moderate
# ---------------------------------------------------------------------------


class TestModerate:

    def test_moderate_returns_result(self, service, monkeypatch):
        mock_result = _mock_mod_result("SAFE")
        monkeypatch.setattr(
            service._moderation_client,
            "moderate",
            lambda **kwargs: mock_result,
        )

        result = service.moderate(
            "Is this safe?",
            model="meta-llama/llama-prompt-guard-2-86m",
        )
        assert result.label == "SAFE"

    @pytest.mark.asyncio
    async def test_async_moderate_returns_result(self, service, monkeypatch):
        mock_result = _mock_mod_result("SAFE")
        monkeypatch.setattr(
            service._moderation_client,
            "async_moderate",
            AsyncMock(return_value=mock_result),
        )

        result = await service.async_moderate(
            "check this",
            model="meta-llama/llama-prompt-guard-2-86m",
        )
        assert result.label == "SAFE"


# ---------------------------------------------------------------------------
# transcribe / async_transcribe / translate / async_translate
# ---------------------------------------------------------------------------


class TestTranscribe:

    def test_transcribe_returns_result(self, service, monkeypatch):
        mock_result = _mock_transcription_result()
        monkeypatch.setattr(
            service._transcription_client,
            "transcribe",
            lambda **kwargs: mock_result,
        )

        result = service.transcribe(b"audio", model="whisper-large-v3")
        assert result.text == "hello world"

    @pytest.mark.asyncio
    async def test_async_transcribe_returns_result(self, service, monkeypatch):
        mock_result = _mock_transcription_result()
        monkeypatch.setattr(
            service._transcription_client,
            "async_transcribe",
            AsyncMock(return_value=mock_result),
        )

        result = await service.async_transcribe(b"audio", model="whisper-large-v3")
        assert result.text == "hello world"

    def test_translate_returns_result(self, service, monkeypatch):
        mock_result = _mock_transcription_result()
        monkeypatch.setattr(
            service._transcription_client,
            "translate",
            lambda **kwargs: mock_result,
        )

        result = service.translate(b"audio", model="whisper-large-v3")
        assert result.text == "hello world"

    @pytest.mark.asyncio
    async def test_async_translate_returns_result(self, service, monkeypatch):
        mock_result = _mock_transcription_result()
        monkeypatch.setattr(
            service._transcription_client,
            "async_translate",
            AsyncMock(return_value=mock_result),
        )

        result = await service.async_translate(b"audio", model="whisper-large-v3")
        assert result.text == "hello world"


# ---------------------------------------------------------------------------
# synthesize / async_synthesize
# ---------------------------------------------------------------------------


class TestSynthesize:

    def test_synthesize_returns_result(self, service, monkeypatch):
        mock_result = _mock_synthesis_result()
        monkeypatch.setattr(
            service._synthesis_client,
            "synthesize",
            lambda **kwargs: mock_result,
        )

        result = service.synthesize(
            "Hello",
            "Aria",
            model="canopylabs/orpheus-v1-english",
        )
        assert result.audio == b"\x00\x01\x02\x03"

    @pytest.mark.asyncio
    async def test_async_synthesize_returns_result(self, service, monkeypatch):
        mock_result = _mock_synthesis_result()
        monkeypatch.setattr(
            service._synthesis_client,
            "async_synthesize",
            AsyncMock(return_value=mock_result),
        )

        result = await service.async_synthesize(
            "Hello",
            "Aria",
            model="canopylabs/orpheus-v1-english",
        )
        assert result.voice == "Aria"


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


class TestSessionHelpers:

    def test_get_session_stats_returns_none_for_unknown_session(self, service):
        result = service.get_session_stats("nonexistent-session")
        assert result is None

    def test_get_session_stats_returns_stats_after_invoke(self, service, monkeypatch):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion("ok")
        monkeypatch.setattr(service._chat_client, "_sync_sdk", lambda api_key: mock_sdk)

        sid = "test-session-stats"
        service.invoke("Q", model="llama-3.3-70b-versatile", session_id=sid)
        stats = service.get_session_stats(sid)
        assert stats is not None

    def test_cleanup_sessions_returns_count(self, service):
        result = service.cleanup_sessions()
        assert isinstance(result, int)
        assert result >= 0


# ---------------------------------------------------------------------------
# structured() with code-fence output
# ---------------------------------------------------------------------------


class TestStructuredCodeFence:

    def test_structured_strips_markdown_code_fence(self, service, monkeypatch):
        class Answer(BaseModel):
            answer: str

        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion(
            '```json\n{"answer": "42"}\n```',
        )
        monkeypatch.setattr(service._structured_client, "_sync_sdk", lambda api_key: mock_sdk)

        result = service.structured("What is 6*7?", Answer, model="llama-3.3-70b-versatile")
        assert result.answer == "42"


# ---------------------------------------------------------------------------
# Error paths in _run_with_rotation (auth error causes immediate break)
# ---------------------------------------------------------------------------


class TestRotationErrorPaths:

    def test_auth_error_breaks_retry_loop_immediately(self, service, monkeypatch):
        call_count = [0]

        def factory(api_key: str) -> MagicMock:
            sdk = MagicMock()
            exc = Exception("auth failure")
            exc.status_code = 401
            sdk.chat.completions.create.side_effect = exc
            call_count[0] += 1
            return sdk

        monkeypatch.setattr(service._chat_client, "_sync_sdk", factory)
        monkeypatch.setattr(service._chat_client, "_handle_sdk_error", lambda *a: None)

        cfg = RequestConfig(retries=3)
        with pytest.raises(RetryExhaustedError):
            service.invoke("Hi", model="llama-3.3-70b-versatile", config=cfg)
        # Auth error should break the loop after one attempt
        assert call_count[0] == 1

    def test_api_key_disabled_error_raises_immediately(self, service, monkeypatch):
        def factory(api_key: str) -> MagicMock:
            sdk = MagicMock()
            sdk.chat.completions.create.side_effect = APIKeyDisabledError(
                key_id="key_0",
                status_code=401,
            )
            return sdk

        monkeypatch.setattr(service._chat_client, "_sync_sdk", factory)

        cfg = RequestConfig(retries=2)
        with pytest.raises(APIKeyDisabledError):
            service.invoke("Hi", model="llama-3.3-70b-versatile", config=cfg)

    @pytest.mark.asyncio
    async def test_async_rotation_exhausted_raises(self, service, monkeypatch):
        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("always fail"),
        )
        monkeypatch.setattr(service._chat_client, "_async_sdk", lambda api_key: mock_sdk)

        cfg = RequestConfig(retries=1)
        with pytest.raises(RetryExhaustedError):
            await service.async_invoke("Hi", model="llama-3.3-70b-versatile", config=cfg)


# ---------------------------------------------------------------------------
# batch() error case
# ---------------------------------------------------------------------------


class TestBatchErrors:

    @pytest.mark.asyncio
    async def test_batch_partial_failure_counted(self, service, monkeypatch):
        call_count = [0]

        async def _create(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("second prompt fails")
            return _mock_completion("ok")

        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = _create
        monkeypatch.setattr(service._chat_client, "_async_sdk", lambda api_key: mock_sdk)

        summary = await service.batch(
            ["p1", "p2", "p3"],
            model="llama-3.3-70b-versatile",
            config=RequestConfig(retries=0),
        )
        assert summary.total == 3
        assert summary.failed >= 1


# ---------------------------------------------------------------------------
# Service init — no-persistence path and EmptyKeyPoolError
# ---------------------------------------------------------------------------


class TestServiceInit:

    def test_service_with_no_base_dir_disables_persistence(self):
        config = GroqConfig(api_keys=["gsk_test"], paths=PathConfig(base_dir=None))
        svc = GroqService(config=config)
        assert svc._usage_persistence is None
        assert svc._token_persistence is None
        assert svc._account_persistence is None

    def test_service_with_no_keys_raises_empty_key_pool(self):
        config = GroqConfig(api_keys=[], paths=PathConfig(base_dir=None))
        with pytest.raises(EmptyKeyPoolError):
            GroqService(config=config)

    def test_service_no_persistence_invoke_covers_journal_early_return(self, monkeypatch):
        """
        With PathConfig(base_dir=None), _journal is None. A successful invoke
        hits _journal_entry() which returns immediately at the 'if self._journal is None'
        guard (line 250).
        """
        config = GroqConfig(api_keys=["gsk_test"], paths=PathConfig(base_dir=None))
        svc = GroqService(config=config)

        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.return_value = _mock_completion("ok")
        monkeypatch.setattr(svc._chat_client, "_sync_sdk", lambda api_key: mock_sdk)

        response = svc.invoke("Hello", model="llama-3.3-70b-versatile")
        assert response.text == "ok"


# ---------------------------------------------------------------------------
# _resolve_session — SessionExpiredError re-raise
# ---------------------------------------------------------------------------


class TestResolveSession:

    def test_expired_session_error_is_reraised(self, service, monkeypatch):
        def _raise(*args, **kwargs):
            raise SessionExpiredError("old-session-id")

        monkeypatch.setattr(service._session_manager, "get_or_create", _raise)
        with pytest.raises(SessionExpiredError):
            service.invoke("hello", model="llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# _run_rotation_generic exhausted path (moderate failing all retries)
# ---------------------------------------------------------------------------


class TestRotationGenericExhausted:

    def test_moderate_all_retries_exhausted_raises(self, service, monkeypatch):
        def _bad_moderate(**kwargs):
            raise RuntimeError("moderation failure")

        monkeypatch.setattr(service._moderation_client, "moderate", _bad_moderate)

        cfg = RequestConfig(retries=1)
        with pytest.raises(RetryExhaustedError):
            service.moderate(
                "text to moderate",
                model="meta-llama/llama-prompt-guard-2-86m",
                config=cfg,
            )

    @pytest.mark.asyncio
    async def test_async_moderate_all_retries_exhausted_raises(self, service, monkeypatch):
        monkeypatch.setattr(
            service._moderation_client,
            "async_moderate",
            AsyncMock(side_effect=RuntimeError("async moderation failure")),
        )

        cfg = RequestConfig(retries=1)
        with pytest.raises(RetryExhaustedError):
            await service.async_moderate(
                "text to moderate",
                model="meta-llama/llama-prompt-guard-2-86m",
                config=cfg,
            )


# ---------------------------------------------------------------------------
# _async_run_with_rotation error paths (APIKeyDisabledError + auth break)
# ---------------------------------------------------------------------------


class TestAsyncRotationErrorPaths:

    @pytest.mark.asyncio
    async def test_async_api_key_disabled_raises_immediately(self, service, monkeypatch):
        async def _create(**kwargs):
            raise APIKeyDisabledError(key_id="key_0", status_code=401)

        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = _create
        monkeypatch.setattr(service._chat_client, "_async_sdk", lambda api_key: mock_sdk)

        cfg = RequestConfig(retries=2)
        with pytest.raises(APIKeyDisabledError):
            await service.async_invoke("Hi", model="llama-3.3-70b-versatile", config=cfg)

    @pytest.mark.asyncio
    async def test_async_auth_error_breaks_loop_immediately(self, service, monkeypatch):
        call_count = [0]

        async def _create(**kwargs):
            call_count[0] += 1
            exc = Exception("async auth failure")
            exc.status_code = 401
            raise exc

        mock_sdk = AsyncMock()
        mock_sdk.chat.completions.create = _create
        monkeypatch.setattr(service._chat_client, "_async_sdk", lambda api_key: mock_sdk)
        monkeypatch.setattr(service._chat_client, "_handle_sdk_error", lambda *a: None)

        cfg = RequestConfig(retries=3)
        with pytest.raises(RetryExhaustedError):
            await service.async_invoke("Hi", model="llama-3.3-70b-versatile", config=cfg)
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# stream() error paths: mid-stream failure and pre-chunk exhaustion
# ---------------------------------------------------------------------------


class TestStreamErrorPaths:

    def test_stream_error_after_first_chunk_raises_immediately(self, service, monkeypatch):
        def _mid_stream_failure(**kwargs):
            yield "first chunk"
            raise RuntimeError("mid-stream failure")

        monkeypatch.setattr(service._chat_client, "stream", _mid_stream_failure)

        cfg = RequestConfig(retries=3)
        chunks = []
        with pytest.raises(RuntimeError, match="mid-stream failure"):
            for chunk in service.stream("Hello", model="llama-3.3-70b-versatile", config=cfg):
                chunks.append(chunk)
        assert "first chunk" in chunks

    def test_stream_exhausted_before_first_chunk_raises(self, service, monkeypatch):
        def _pre_chunk_failure(**kwargs):
            raise RuntimeError("pre-chunk failure")
            yield  # noqa: unreachable — makes this a generator

        monkeypatch.setattr(service._chat_client, "stream", _pre_chunk_failure)

        cfg = RequestConfig(retries=1)
        with pytest.raises(RetryExhaustedError):
            list(service.stream("Hello", model="llama-3.3-70b-versatile", config=cfg))

    @pytest.mark.asyncio
    async def test_async_stream_exhausted_before_first_chunk_raises(self, service, monkeypatch):
        async def _pre_chunk_failure(**kwargs):
            raise RuntimeError("pre-chunk async failure")
            yield  # noqa: unreachable — makes this an async generator

        monkeypatch.setattr(service._chat_client, "async_stream", _pre_chunk_failure)

        cfg = RequestConfig(retries=1)
        with pytest.raises(RetryExhaustedError):
            async for _ in service.async_stream(
                "Hello",
                model="llama-3.3-70b-versatile",
                config=cfg,
            ):
                pass
