"""
Retry policies built on tenacity.

Retryable errors:    429, 5xx, timeouts, connection issues
Non-retryable:       401, 403, 400, invalid model, malformed request
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, TypeVar

from tenacity import (
	AsyncRetrying,
	RetryCallState,
	Retrying,
	retry_if_exception,
	stop_after_attempt,
	wait_exponential,
	wait_random,
	)

from exceptions.keys import APIKeyDisabledError, NoAvailableAPIKeyError
from exceptions.output import StructuredOutputError
from exceptions.response import InvalidResponseError


T = TypeVar("T")


class ErrorCategory(str, Enum):
	"""High-level retry classification used by RetryClassifier."""

	TRANSPORT = "transport"
	PROVIDER = "provider"
	QUOTA = "quota"
	VALIDATION = "validation"
	UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_FATAL_STATUS_CODES = {400, 401, 403, 404, 422}


def _is_retryable(exc: BaseException) -> bool:
	"""Return True when the exception warrants another attempt."""
	# Never retry these
	if isinstance(exc, (APIKeyDisabledError, StructuredOutputError, NoAvailableAPIKeyError)):
		return False

	if isinstance(exc, InvalidResponseError):
		code = getattr(exc, "status_code", None)
		if code in _FATAL_STATUS_CODES:
			return False
		return code in _RETRYABLE_STATUS_CODES

	# groq SDK exceptions — inspect the message/type
	exc_type = type(exc).__name__
	retryable_names = {
		"RateLimitError",
		"APIStatusError",
		"APIConnectionError",
		"APITimeoutError",
		"InternalServerError",
		"ServiceUnavailableError",
		}
	if exc_type in retryable_names:
		# 401/403 → not retryable even for SDK types
		status = getattr(exc, "status_code", None)
		return status not in _FATAL_STATUS_CODES

	# httpx / network errors
	import httpx

	return isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError))


def _is_rate_limit(exc: BaseException) -> bool:
	exc_type = type(exc).__name__
	if exc_type == "RateLimitError":
		return True
	return isinstance(exc, InvalidResponseError) and getattr(exc, "status_code", None) == 429


def _is_auth_error(exc: BaseException) -> bool:
	status = getattr(exc, "status_code", None)
	return status in (401, 403) or type(exc).__name__ in (
		"AuthenticationError",
		"PermissionDeniedError",
		)


class RetryClassifier:
	"""Classify provider/client exceptions and decide whether they can be retried."""

	def classify(self, exc: BaseException) -> ErrorCategory:
		if isinstance(exc, (APIKeyDisabledError, NoAvailableAPIKeyError)):
			return ErrorCategory.QUOTA
		if isinstance(exc, StructuredOutputError):
			return ErrorCategory.VALIDATION
		if isinstance(exc, InvalidResponseError):
			status_code = getattr(exc, "status_code", None)
			if status_code == 429:
				return ErrorCategory.QUOTA
			if status_code in _FATAL_STATUS_CODES:
				return ErrorCategory.VALIDATION
			if status_code in _RETRYABLE_STATUS_CODES:
				return ErrorCategory.PROVIDER
			return ErrorCategory.UNKNOWN

		exc_type = type(exc).__name__
		if exc_type in {
			"APIConnectionError",
			"APITimeoutError",
			"TimeoutException",
			"ConnectError",
			"RemoteProtocolError",
			}:
			return ErrorCategory.TRANSPORT
		if exc_type in {"RateLimitError"}:
			return ErrorCategory.QUOTA
		if exc_type in {"APIStatusError", "InternalServerError", "ServiceUnavailableError"}:
			return ErrorCategory.PROVIDER
		if exc_type in {
			"AuthenticationError",
			"PermissionDeniedError",
			"BadRequestError",
			"NotFoundError",
			}:
			return ErrorCategory.VALIDATION
		return ErrorCategory.UNKNOWN

	def is_retriable(self, exc: BaseException) -> bool:
		return _is_retryable(exc)


class BackoffCalculator:
	"""Exponential backoff calculator with optional bounded jitter."""

	def __init__(
			self, base_backoff: float = 1.0, max_backoff: float = 30.0, jitter: float = 0.5,
			) -> None:
		self.base_backoff = base_backoff
		self.max_backoff = max_backoff
		self.jitter = jitter

	def calculate(self, attempt: int) -> float:
		delay = min(self.max_backoff, self.base_backoff * (2**max(0, attempt - 1)))
		if self.jitter <= 0:
			return delay
		return delay + random.uniform(0, self.jitter)


# ---------------------------------------------------------------------------
# Sync retry policy
# ---------------------------------------------------------------------------


class RetryPolicy:
	"""
	Wraps tenacity Retrying.
	Exposes a simple `execute(fn, *args, **kwargs)` interface.
	"""

	def __init__(
			self,
			max_retries: int = 3,
			base_backoff: float = 1.0,
			max_backoff: float = 30.0,
			on_retry: Callable[[RetryCallState], None] | None = None,
			) -> None:
		self._max_retries = max_retries
		self._base_backoff = base_backoff
		self._max_backoff = max_backoff
		self._on_retry = on_retry

	def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
		retryer = Retrying(
			retry=retry_if_exception(_is_retryable),
			stop=stop_after_attempt(self._max_retries + 1),
			wait=wait_exponential(multiplier=self._base_backoff, max=self._max_backoff)
			     + wait_random(0, 0.5),
			after=self._on_retry,
			reraise=True,
			)
		return retryer(fn, *args, **kwargs)


# ---------------------------------------------------------------------------
# Async retry policy
# ---------------------------------------------------------------------------


class AsyncRetryPolicy:
	"""
	Wraps tenacity AsyncRetrying.
	"""

	def __init__(
			self,
			max_retries: int = 3,
			base_backoff: float = 1.0,
			max_backoff: float = 30.0,
			on_retry: Callable[[RetryCallState], None] | None = None,
			) -> None:
		self._max_retries = max_retries
		self._base_backoff = base_backoff
		self._max_backoff = max_backoff
		self._on_retry = on_retry

	async def execute(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
		retryer = AsyncRetrying(
			retry=retry_if_exception(_is_retryable),
			stop=stop_after_attempt(self._max_retries + 1),
			wait=wait_exponential(multiplier=self._base_backoff, max=self._max_backoff)
			     + wait_random(0, 0.5),
			after=self._on_retry,
			reraise=True,
			)
		return await retryer(fn, *args, **kwargs)


class RetryExecutor:
	"""Spec-compatible facade exposing execute()/aexecute() over retry policies."""

	def __init__(
			self,
			max_retries: int = 3,
			base_backoff: float = 1.0,
			max_backoff: float = 30.0,
			on_retry: Callable[[RetryCallState], None] | None = None,
			) -> None:
		self._sync = RetryPolicy(max_retries, base_backoff, max_backoff, on_retry)
		self._async = AsyncRetryPolicy(max_retries, base_backoff, max_backoff, on_retry)

	def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
		return self._sync.execute(fn, *args, **kwargs)

	async def aexecute(self, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
		return await self._async.execute(fn, *args, **kwargs)
