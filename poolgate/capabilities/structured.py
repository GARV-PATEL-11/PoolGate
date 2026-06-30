"""StructuredCapability — JSON-mode / schema-constrained generation."""

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
from poolgate.providers.groq.capabilities import StructuredGenerationCapability
from poolgate.providers.groq.client import GroqProvider
from poolgate.schemas.common.runtime import GroqResponse, RequestConfig


def _build_response_format(json_schema: dict[str, Any] | None) -> dict[str, Any]:
    if json_schema is None:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": json_schema.get("title", "response"),
            "schema": json_schema,
            "strict": True,
        },
    }


class StructuredCapability(GroqProvider, StructuredGenerationCapability):
    """Stateless client for JSON-mode / schema-constrained generation."""

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
        rid = _new_rid(request_id)
        client = self._sync_sdk(api_key)
        response_format = _build_response_format(json_schema)
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
                response_format=response_format,  # type: ignore[arg-type]
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
        rid = _new_rid(request_id)
        client = self._async_sdk(api_key)
        response_format = _build_response_format(json_schema)
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
                response_format=response_format,  # type: ignore[arg-type]
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
