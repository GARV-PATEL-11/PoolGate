"""GeminiStructuredCapability — JSON-schema-constrained generation via Google Gemini."""

from __future__ import annotations

import time
from typing import Any

from poolgate.providers.gemini.capabilities import GeminiStructuredGenerationCapability
from poolgate.providers.gemini.client import (
    GeminiProvider,
    _convert_messages,
    _new_rid,
    _parse_gemini_finish_reason,
    _parse_gemini_text,
    _parse_gemini_usage,
)
from poolgate.schemas.common.runtime import GroqResponse, RequestConfig


class GeminiStructuredCapability(GeminiProvider, GeminiStructuredGenerationCapability):
    """Stateless structured-output client for Google Gemini."""

    def invoke_structured(
        self,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        config: RequestConfig,
        session_id: str,
        api_key_id: str,
        json_schema: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> GroqResponse:
        from google.genai import types  # type: ignore[import-untyped]

        rid = _new_rid(request_id)
        client = self._sync_sdk(api_key)
        system_instruction, contents = _convert_messages(messages)
        gen_kwargs: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "response_mime_type": "application/json",
        }
        if config.max_tokens is not None:
            gen_kwargs["max_output_tokens"] = config.max_tokens
        if system_instruction is not None:
            gen_kwargs["system_instruction"] = system_instruction
        if json_schema is not None:
            gen_kwargs["response_schema"] = json_schema
        start = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**gen_kwargs),
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

    async def async_invoke_structured(
        self,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        config: RequestConfig,
        session_id: str,
        api_key_id: str,
        json_schema: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> GroqResponse:
        from google.genai import types  # type: ignore[import-untyped]

        rid = _new_rid(request_id)
        client = self._async_sdk(api_key)
        system_instruction, contents = _convert_messages(messages)
        gen_kwargs: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "response_mime_type": "application/json",
        }
        if config.max_tokens is not None:
            gen_kwargs["max_output_tokens"] = config.max_tokens
        if system_instruction is not None:
            gen_kwargs["system_instruction"] = system_instruction
        if json_schema is not None:
            gen_kwargs["response_schema"] = json_schema
        start = time.perf_counter()
        try:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**gen_kwargs),
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
