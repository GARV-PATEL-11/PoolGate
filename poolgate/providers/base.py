"""BaseProvider — LSP guarantee for all provider adapters."""

from __future__ import annotations

import contextlib
import uuid
from abc import ABC
from typing import Any

from groq import AsyncGroq, Groq
from groq.types.chat import ChatCompletion

from poolgate.exceptions.keys import APIKeyDisabledError
from poolgate.exceptions.rate_limit import RateLimitExceededError
from poolgate.exceptions.response import InvalidResponseError
from poolgate.schemas.common.runtime import FinishReason
from poolgate.schemas.responses.usage import TokenUsage
from poolgate.services.retry import is_auth_error, is_rate_limit


def _parse_finish_reason(raw: str | None) -> FinishReason:
    mapping = {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
    }
    return mapping.get(raw or "", FinishReason.UNKNOWN)


def _parse_usage(completion: ChatCompletion) -> TokenUsage:
    usage = completion.usage
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        total_tokens=usage.total_tokens or 0,
    )


def _parse_chunk_usage(chunk: Any) -> TokenUsage | None:
    x_groq = getattr(chunk, "x_groq", None)
    if x_groq is None:
        return None
    usage = getattr(x_groq, "usage", None)
    if usage is None:
        return None
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


def _new_rid(request_id: str | None) -> str:
    return request_id or str(uuid.uuid4())


def _first_choice(completion: Any, request_id: str) -> Any:
    try:
        return completion.choices[0]
    except (AttributeError, IndexError, TypeError) as exc:
        raise InvalidResponseError(
            "Groq returned a completion without choices[0].",
            status_code=getattr(completion, "status_code", None),
            raw_response=completion,
            request_id=request_id,
        ) from exc


def _choice_text(choice: Any, request_id: str) -> str:
    try:
        return choice.message.content or ""
    except AttributeError as exc:
        raise InvalidResponseError(
            "Groq returned a choice without message.content.",
            raw_response=choice,
            request_id=request_id,
        ) from exc


def _chunk_delta_text(chunk: Any, request_id: str) -> str:
    try:
        return chunk.choices[0].delta.content or ""
    except (AttributeError, IndexError, TypeError) as exc:
        raise InvalidResponseError(
            "Groq returned a malformed streaming chunk.",
            raw_response=chunk,
            request_id=request_id,
        ) from exc


class BaseProvider(ABC):
    """Abstract base for all provider adapters. Provides SDK construction and error mapping."""

    @staticmethod
    def _sync_sdk(api_key: str) -> Groq:
        return Groq(api_key=api_key)

    @staticmethod
    def _async_sdk(api_key: str) -> AsyncGroq:
        return AsyncGroq(api_key=api_key)

    def _handle_sdk_error(self, exc: Exception, request_id: str, api_key_id: str = "unknown") -> None:
        if is_auth_error(exc):
            raise APIKeyDisabledError(
                key_id=api_key_id,
                status_code=getattr(exc, "status_code", None),
                request_id=request_id,
            ) from exc

        if is_rate_limit(exc):
            retry_after: float = 60.0
            response = getattr(exc, "response", None)
            if response:
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


# Re-export parsing helpers for capability clients
__all__ = [
    "BaseProvider",
    "_parse_finish_reason",
    "_parse_usage",
    "_parse_chunk_usage",
    "_new_rid",
    "_first_choice",
    "_choice_text",
    "_chunk_delta_text",
]
