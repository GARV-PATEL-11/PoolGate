"""GeminiToolCapability — function/tool calling via Google Gemini."""

from __future__ import annotations

import time
from typing import Any

from poolgate.providers.gemini.capabilities import GeminiToolCallingCapability
from poolgate.providers.gemini.client import (
    GeminiProvider,
    _convert_messages,
    _convert_tools,
    _extract_tool_calls,
    _new_rid,
    _parse_gemini_finish_reason,
    _parse_gemini_text,
    _parse_gemini_usage,
)
from poolgate.schemas.common.runtime import FinishReason, GroqResponse, RequestConfig


class GeminiToolCapability(GeminiProvider, GeminiToolCallingCapability):
    """Stateless function/tool calling client for Google Gemini."""

    def invoke_tools(
        self,
        api_key: str,
        model: str,
        messages: list[dict[str, Any]],
        config: RequestConfig,
        session_id: str,
        api_key_id: str,
        tools: list[dict[str, Any]],
        request_id: str | None = None,
    ) -> GroqResponse:
        from google.genai import types  # type: ignore[import-untyped]

        rid = _new_rid(request_id)
        client = self._sync_sdk(api_key)
        system_instruction, contents = _convert_messages([{k: str(v) for k, v in m.items()} for m in messages])
        gemini_tools = _convert_tools(tools)
        gen_kwargs: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "tools": gemini_tools,
        }
        if config.max_tokens is not None:
            gen_kwargs["max_output_tokens"] = config.max_tokens
        if system_instruction is not None:
            gen_kwargs["system_instruction"] = system_instruction
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
        tool_calls = _extract_tool_calls(response)
        finish_reason = FinishReason.TOOL_CALLS if tool_calls else _parse_gemini_finish_reason(response)
        text = ""
        try:
            text = response.text or ""
        except (AttributeError, ValueError):
            pass
        return GroqResponse(
            text=text,
            model=model,
            usage=_parse_gemini_usage(response),
            latency=latency,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
            finish_reason=finish_reason,
            metadata={"tool_calls": tool_calls},
            raw_response=response,
        )

    async def async_invoke_tools(
        self,
        api_key: str,
        model: str,
        messages: list[dict[str, Any]],
        config: RequestConfig,
        session_id: str,
        api_key_id: str,
        tools: list[dict[str, Any]],
        request_id: str | None = None,
    ) -> GroqResponse:
        from google.genai import types  # type: ignore[import-untyped]

        rid = _new_rid(request_id)
        client = self._async_sdk(api_key)
        system_instruction, contents = _convert_messages([{k: str(v) for k, v in m.items()} for m in messages])
        gemini_tools = _convert_tools(tools)
        gen_kwargs: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "tools": gemini_tools,
        }
        if config.max_tokens is not None:
            gen_kwargs["max_output_tokens"] = config.max_tokens
        if system_instruction is not None:
            gen_kwargs["system_instruction"] = system_instruction
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
        tool_calls = _extract_tool_calls(response)
        finish_reason = FinishReason.TOOL_CALLS if tool_calls else _parse_gemini_finish_reason(response)
        text = ""
        try:
            text = response.text or ""
        except (AttributeError, ValueError):
            pass
        return GroqResponse(
            text=text,
            model=model,
            usage=_parse_gemini_usage(response),
            latency=latency,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
            finish_reason=finish_reason,
            metadata={"tool_calls": tool_calls},
            raw_response=response,
        )
