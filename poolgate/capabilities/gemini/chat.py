"""GeminiChatCapability — text generation via Google Gemini."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any

from poolgate.providers.gemini.capabilities import GeminiTextGenerationCapability
from poolgate.providers.gemini.client import (
    GeminiProvider,
    _convert_messages,
    _new_rid,
    _parse_gemini_finish_reason,
    _parse_gemini_text,
    _parse_gemini_usage,
)
from poolgate.schemas.common.runtime import GroqResponse, RequestConfig, TokenUsage


def _build_gen_config(config: RequestConfig, system_instruction: str | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    if config.max_tokens is not None:
        kwargs["max_output_tokens"] = config.max_tokens
    if config.stop is not None:
        stops = config.stop if isinstance(config.stop, list) else [config.stop]
        kwargs["stop_sequences"] = stops
    if system_instruction is not None:
        kwargs["system_instruction"] = system_instruction
    return kwargs


class GeminiChatCapability(GeminiProvider, GeminiTextGenerationCapability):
    """Stateless chat completion client for Google Gemini."""

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
        from google.genai import types  # type: ignore[import-untyped]

        rid = _new_rid(request_id)
        client = self._sync_sdk(api_key)
        system_instruction, contents = _convert_messages(messages)
        start = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**_build_gen_config(config, system_instruction)),
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise
        latency = time.perf_counter() - start
        return GroqResponse(
            text=_parse_gemini_text(response, rid),
            model=model,
            usage=_parse_gemini_usage(response),
            latency=latency,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
            finish_reason=_parse_gemini_finish_reason(response),
            raw_response=response,
        )

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
        from google.genai import types  # type: ignore[import-untyped]

        rid = _new_rid(request_id)
        client = self._async_sdk(api_key)
        system_instruction, contents = _convert_messages(messages)
        start = time.perf_counter()
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**_build_gen_config(config, system_instruction)),
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise
        latency = time.perf_counter() - start
        return GroqResponse(
            text=_parse_gemini_text(response, rid),
            model=model,
            usage=_parse_gemini_usage(response),
            latency=latency,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
            finish_reason=_parse_gemini_finish_reason(response),
            raw_response=response,
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
        from google.genai import types  # type: ignore[import-untyped]

        rid = _new_rid(request_id)
        client = self._sync_sdk(api_key)
        system_instruction, contents = _convert_messages(messages)
        try:
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**_build_gen_config(config, system_instruction)),
            ):
                if on_usage is not None:
                    usage = _parse_gemini_usage(chunk)
                    if usage.total_tokens > 0:
                        on_usage(usage)
                text = getattr(chunk, "text", "") or ""
                if text:
                    yield text
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise

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
        from google.genai import types  # type: ignore[import-untyped]

        rid = _new_rid(request_id)
        client = self._async_sdk(api_key)
        system_instruction, contents = _convert_messages(messages)
        try:
            async for chunk in client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**_build_gen_config(config, system_instruction)),
            ):
                if on_usage is not None:
                    usage = _parse_gemini_usage(chunk)
                    if usage.total_tokens > 0:
                        on_usage(usage)
                text = getattr(chunk, "text", "") or ""
                if text:
                    yield text
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise
