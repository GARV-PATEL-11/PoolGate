"""ChatCapability — text generation (chat completion)."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Callable, Generator

from poolgate.providers.base import (
    _choice_text,
    _chunk_delta_text,
    _first_choice,
    _new_rid,
    _parse_chunk_usage,
    _parse_finish_reason,
    _parse_usage,
)
from poolgate.providers.groq.capabilities import TextGenerationCapability
from poolgate.providers.groq.client import GroqProvider
from poolgate.schemas.common.runtime import GroqResponse, RequestConfig, TokenUsage


class ChatCapability(GroqProvider, TextGenerationCapability):
    """Stateless client for text generation (chat completion)."""

    def invoke(
        self,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        config: RequestConfig,
        session_id: str,
        api_key_id: str,
        request_id: str | None = None,
    ) -> GroqResponse:
        rid = _new_rid(request_id)
        client = self._sync_sdk(api_key)
        start = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                seed=config.seed,
                stop=config.stop,
                timeout=config.timeout,
                stream=False,
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise
        choice = _first_choice(completion, rid)
        latency = time.perf_counter() - start
        return GroqResponse(
            text=_choice_text(choice, rid),
            model=model,
            usage=_parse_usage(completion),
            latency=latency,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
            finish_reason=_parse_finish_reason(getattr(choice, "finish_reason", None)),
            raw_response=completion,
        )

    def stream(
        self,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        config: RequestConfig,
        session_id: str,
        api_key_id: str,
        request_id: str | None = None,
        on_usage: Callable[[TokenUsage], None] | None = None,
    ) -> Generator[str, None, None]:
        rid = _new_rid(request_id)
        client = self._sync_sdk(api_key)
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                seed=config.seed,
                stop=config.stop,
                timeout=config.timeout,
                stream=True,
            )
            with stream as s:
                for chunk in s:
                    usage = _parse_chunk_usage(chunk)
                    if usage is not None and on_usage is not None:
                        on_usage(usage)
                    delta = _chunk_delta_text(chunk, rid)
                    if delta:
                        yield delta
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise

    async def async_invoke(
        self,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        config: RequestConfig,
        session_id: str,
        api_key_id: str,
        request_id: str | None = None,
    ) -> GroqResponse:
        rid = _new_rid(request_id)
        client = self._async_sdk(api_key)
        start = time.perf_counter()
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                seed=config.seed,
                stop=config.stop,
                timeout=config.timeout,
                stream=False,
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise
        choice = _first_choice(completion, rid)
        latency = time.perf_counter() - start
        return GroqResponse(
            text=_choice_text(choice, rid),
            model=model,
            usage=_parse_usage(completion),
            latency=latency,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
            finish_reason=_parse_finish_reason(getattr(choice, "finish_reason", None)),
            raw_response=completion,
        )

    async def async_stream(
        self,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        config: RequestConfig,
        session_id: str,
        api_key_id: str,
        request_id: str | None = None,
        on_usage: Callable[[TokenUsage], None] | None = None,
    ) -> AsyncGenerator[str, None]:
        rid = _new_rid(request_id)
        client = self._async_sdk(api_key)
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                seed=config.seed,
                stop=config.stop,
                timeout=config.timeout,
                stream=True,
            )
            async with stream as s:
                async for chunk in s:
                    usage = _parse_chunk_usage(chunk)
                    if usage is not None and on_usage is not None:
                        on_usage(usage)
                    delta = _chunk_delta_text(chunk, rid)
                    if delta:
                        yield delta
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise
