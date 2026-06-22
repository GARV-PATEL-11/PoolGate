"""
base.py — Shared internals for all PoolGate capability clients.

Every concrete client inherits BaseGroqClient which provides:
  - sync / async Groq SDK constructors (_sync_sdk / _async_sdk)
  - unified SDK error → structured exception mapping (_handle_sdk_error)
  - finish-reason and usage parsing helpers (module-level functions)

Nothing in here is model-specific; it is pure infrastructure.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Any

from groq import AsyncGroq, Groq
from groq.types.chat import ChatCompletion

from exceptions.keys import APIKeyDisabledError
from exceptions.rate_limit import RateLimitExceededError
from exceptions.response import InvalidResponseError
from retry import _is_auth_error, _is_rate_limit
from schemas.runtime import FinishReason, TokenUsage


# ---------------------------------------------------------------------------
# Module-level parsing helpers (importable by all capability clients)
# ---------------------------------------------------------------------------


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
	"""
	Extract token usage from a streaming ChatCompletionChunk, if present.

	The Groq SDK does not support the OpenAI-style `stream_options:
	{"include_usage": true}` request parameter (verified against the
	groq==1.4.0 SDK's completion_create_params — the field does not exist
	there). Instead Groq attaches usage as a vendor extension on the final
	chunk only, at chunk.x_groq.usage. Every other chunk has x_groq=None or
	x_groq.usage=None, so this returns None until the final chunk arrives.
	"""
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
	"""Return the provided request ID or generate a fresh UUID4."""
	return request_id or str(uuid.uuid4())


def _first_choice(completion: Any, request_id: str) -> Any:
	"""Return choices[0] or raise a PoolGate response error."""
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


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseGroqClient:
	"""
	Mixin inherited by every capability client.

	Provides three things only — SDK construction, error normalisation, and
	the parsing helpers above.  Business logic lives entirely in the concrete
	capability subclasses.
	"""

	# ------------------------------------------------------------------
	# SDK construction
	# ------------------------------------------------------------------
	@staticmethod
	def _sync_sdk(api_key: str) -> Groq:
		"""Return a fresh sync Groq client bound to the given key."""
		return Groq(api_key=api_key)

	@staticmethod
	def _async_sdk(api_key: str) -> AsyncGroq:
		"""Return a fresh async Groq client bound to the given key."""
		return AsyncGroq(api_key=api_key)

	# ------------------------------------------------------------------
	# Error normalisation
	# ------------------------------------------------------------------

	def _handle_sdk_error(
			self, exc: Exception, request_id: str, api_key_id: str = "unknown",
			) -> None:
		"""
		Map raw Groq SDK exceptions to structured PoolGate exceptions.

		Auth errors   → APIKeyDisabledError
		Rate limits   → RateLimitExceededError (with retry-after extracted from headers)
		Everything else is re-raised as-is so callers can decide.
		"""
		if _is_auth_error(exc):
			raise APIKeyDisabledError(
				key_id=api_key_id,
				status_code=getattr(exc, "status_code", None),
				request_id=request_id,
				) from exc

		if _is_rate_limit(exc):
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
