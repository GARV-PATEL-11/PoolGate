"""GeminiProvider — base class for all Google Gemini capability adapters."""

from __future__ import annotations

import contextlib
import json
import uuid
from abc import ABC
from typing import Any

from poolgate.exceptions.keys import APIKeyDisabledError
from poolgate.exceptions.rate_limit import RateLimitExceededError
from poolgate.exceptions.response import InvalidResponseError
from poolgate.schemas.common.runtime import FinishReason
from poolgate.schemas.responses.usage import TokenUsage


def _parse_gemini_finish_reason(response: Any) -> FinishReason:
    try:
        reason = str(response.candidates[0].finish_reason).upper()
        if "STOP" in reason:
            return FinishReason.STOP
        if "MAX_TOKENS" in reason:
            return FinishReason.LENGTH
        if "SAFETY" in reason or "RECITATION" in reason:
            return FinishReason.CONTENT_FILTER
        if "TOOL" in reason or "FUNCTION_CALL" in reason:
            return FinishReason.TOOL_CALLS
    except (AttributeError, IndexError):
        pass
    return FinishReason.UNKNOWN


def _parse_gemini_usage(response: Any) -> TokenUsage:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
        completion_tokens=getattr(meta, "candidates_token_count", 0) or 0,
        total_tokens=getattr(meta, "total_token_count", 0) or 0,
    )


def _parse_gemini_text(response: Any, request_id: str) -> str:
    try:
        return response.text or ""
    except (AttributeError, ValueError) as exc:
        raise InvalidResponseError(
            "Gemini returned a response without usable text content.",
            raw_response=response,
            request_id=request_id,
        ) from exc


def _convert_messages(messages: list[dict[str, str]]) -> tuple[str | None, list[Any]]:
    """Convert OpenAI-style messages to Gemini Content format.

    Returns (system_instruction, contents_list). System role becomes a
    top-level instruction; user/assistant become Content objects.
    """
    from google.genai import types  # type: ignore[import-untyped]

    system_instruction: str | None = None
    contents: list[Any] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_instruction = content
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
        else:
            contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
    return system_instruction, contents


def _json_type_to_gemini(json_type: str) -> Any:
    from google.genai import types  # type: ignore[import-untyped]

    return {
        "string": types.Type.STRING,
        "number": types.Type.NUMBER,
        "integer": types.Type.INTEGER,
        "boolean": types.Type.BOOLEAN,
        "object": types.Type.OBJECT,
        "array": types.Type.ARRAY,
    }.get(json_type.lower(), types.Type.STRING)


def _json_schema_to_gemini(schema: dict[str, Any]) -> Any:
    from google.genai import types  # type: ignore[import-untyped]

    json_type = schema.get("type", "string")
    kwargs: dict[str, Any] = {"type": _json_type_to_gemini(json_type)}
    if "description" in schema:
        kwargs["description"] = schema["description"]
    if "enum" in schema:
        kwargs["enum"] = schema["enum"]
    if json_type == "object":
        props = {k: _json_schema_to_gemini(v) for k, v in schema.get("properties", {}).items()}
        if props:
            kwargs["properties"] = props
        if "required" in schema:
            kwargs["required"] = schema["required"]
    elif json_type == "array" and "items" in schema:
        kwargs["items"] = _json_schema_to_gemini(schema["items"])
    return types.Schema(**kwargs)


def _convert_tools(tools: list[dict[str, Any]]) -> list[Any]:
    """Convert OpenAI-style tool dicts to a Gemini Tool list."""
    from google.genai import types  # type: ignore[import-untyped]

    func_decls: list[Any] = []
    for tool in tools:
        fn = tool.get("function", {})
        params = fn.get("parameters", {})
        decl_kwargs: dict[str, Any] = {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
        }
        if params:
            decl_kwargs["parameters"] = _json_schema_to_gemini(params)
        func_decls.append(types.FunctionDeclaration(**decl_kwargs))
    return [types.Tool(function_declarations=func_decls)]


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract Gemini function calls as OpenAI-style tool call dicts."""
    tool_calls: list[dict[str, Any]] = []
    try:
        for part in response.candidates[0].content.parts:
            fc = getattr(part, "function_call", None)
            if fc is None:
                continue
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(dict(fc.args)),
                    },
                }
            )
    except (AttributeError, IndexError):
        pass
    return tool_calls


def _new_rid(request_id: str | None) -> str:
    return request_id or str(uuid.uuid4())


class GeminiProvider(ABC):
    """Abstract base for all Gemini capability adapters. Provides SDK construction and error mapping."""

    @staticmethod
    def _sync_sdk(api_key: str) -> Any:
        from google import genai  # type: ignore[import-untyped]

        return genai.Client(api_key=api_key)

    @staticmethod
    def _async_sdk(api_key: str) -> Any:
        # google-genai uses the same Client for both sync and async;
        # async operations go through client.aio.*
        from google import genai  # type: ignore[import-untyped]

        return genai.Client(api_key=api_key)

    def _handle_sdk_error(self, exc: Exception, request_id: str, api_key_id: str = "unknown") -> None:
        exc_type = type(exc).__name__
        if exc_type in ("PermissionDenied", "Unauthenticated", "Forbidden"):
            raw_code: Any = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            status: int = int(raw_code) if raw_code is not None else 403
            raise APIKeyDisabledError(
                key_id=api_key_id,
                status_code=status,
                request_id=request_id,
            ) from exc
        if exc_type in ("ResourceExhausted", "TooManyRequests"):
            retry_after: float = 60.0
            response = getattr(exc, "response", None)
            if response is not None:
                header = getattr(response, "headers", {}).get("retry-after")
                if header:
                    with contextlib.suppress(TypeError, ValueError):
                        retry_after = float(header)
            raise RateLimitExceededError(
                message=str(exc),
                retry_after=retry_after,
                request_id=request_id,
            ) from exc
        raise exc


__all__ = [
    "GeminiProvider",
    "_convert_messages",
    "_convert_tools",
    "_extract_tool_calls",
    "_json_schema_to_gemini",
    "_new_rid",
    "_parse_gemini_finish_reason",
    "_parse_gemini_text",
    "_parse_gemini_usage",
]
