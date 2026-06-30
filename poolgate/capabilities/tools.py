"""ToolCapability — function/tool calling."""

from __future__ import annotations

import time
from typing import Any

from poolgate.providers.base import (
    _choice_text,
    _first_choice,
    _new_rid,
    _parse_finish_reason,
    _parse_usage,
)
from poolgate.providers.groq.capabilities import ToolCallingCapability
from poolgate.providers.groq.client import GroqProvider
from poolgate.schemas.common.runtime import GroqResponse, RequestConfig


class ToolCapability(GroqProvider, ToolCallingCapability):
    """Stateless client for function/tool calling."""

    def invoke_tools(
        self,
        api_key: str,
        model: str,
        messages: list[dict[str, Any]],
        config: RequestConfig,
        session_id: str,
        api_key_id: str,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
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
                tools=tools,  # type: ignore[arg-type]
                tool_choice=tool_choice,  # type: ignore[arg-type]
                stream=False,
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise
        latency = time.perf_counter() - start
        choice = _first_choice(completion, rid)
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

    async def async_invoke_tools(
        self,
        api_key: str,
        model: str,
        messages: list[dict[str, Any]],
        config: RequestConfig,
        session_id: str,
        api_key_id: str,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
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
                tools=tools,  # type: ignore[arg-type]
                tool_choice=tool_choice,  # type: ignore[arg-type]
                stream=False,
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise
        latency = time.perf_counter() - start
        choice = _first_choice(completion, rid)
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
