"""Retry policies built on tenacity."""

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

from poolgate.exceptions.keys import APIKeyDisabledError, NoAvailableAPIKeyError
from poolgate.exceptions.output import StructuredOutputError
from poolgate.exceptions.response import InvalidResponseError

T = TypeVar("T")

_NOOP_AFTER: Callable[[RetryCallState], None] = lambda _: None


class ErrorCategory(str, Enum):
    TRANSPORT = "transport"
    PROVIDER = "provider"
    QUOTA = "quota"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_FATAL_STATUS_CODES = {400, 401, 403, 404, 422}


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (APIKeyDisabledError, StructuredOutputError, NoAvailableAPIKeyError)):
        return False

    if isinstance(exc, InvalidResponseError):
        code = getattr(exc, "status_code", None)
        if code in _FATAL_STATUS_CODES:
            return False
        return code in _RETRYABLE_STATUS_CODES

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
        status = getattr(exc, "status_code", None)
        return status not in _FATAL_STATUS_CODES

    import httpx

    return isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError))


def is_rate_limit(exc: BaseException) -> bool:
    """Public: return True when the exception is a rate-limit (429-class) error."""
    exc_type = type(exc).__name__
    if exc_type in ("RateLimitError", "RateLimitExceededError"):
        return True
    return isinstance(exc, InvalidResponseError) and getattr(exc, "status_code", None) == 429


def is_auth_error(exc: BaseException) -> bool:
    """Public: return True when the exception indicates an invalid/disabled API key."""
    status = getattr(exc, "status_code", None)
    return status in (401, 403) or type(exc).__name__ in ("AuthenticationError", "PermissionDeniedError")


# Private aliases kept for internal callers that used to import them
_is_rate_limit = is_rate_limit
_is_auth_error = is_auth_error


class RetryClassifier:
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
        if exc_type in {"AuthenticationError", "PermissionDeniedError", "BadRequestError", "NotFoundError"}:
            return ErrorCategory.VALIDATION
        return ErrorCategory.UNKNOWN

    def is_retriable(self, exc: BaseException) -> bool:
        return _is_retryable(exc)


class BackoffCalculator:
    def __init__(self, base_backoff: float = 1.0, max_backoff: float = 30.0, jitter: float = 0.5) -> None:
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.jitter = jitter

    def calculate(self, attempt: int) -> float:
        delay: float = min(self.max_backoff, self.base_backoff * (2 ** max(0, attempt - 1)))
        if self.jitter <= 0:
            return delay
        return delay + random.uniform(0, self.jitter)


class RetryPolicy:
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
            wait=wait_exponential(multiplier=self._base_backoff, max=self._max_backoff) + wait_random(0, 0.5),
            reraise=True,
            after=self._on_retry if self._on_retry is not None else _NOOP_AFTER,
        )
        return retryer(fn, *args, **kwargs)


class AsyncRetryPolicy:
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
            wait=wait_exponential(multiplier=self._base_backoff, max=self._max_backoff) + wait_random(0, 0.5),
            reraise=True,
            after=self._on_retry if self._on_retry is not None else _NOOP_AFTER,
        )
        return await retryer(fn, *args, **kwargs)


class RetryExecutor:
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
        result: T = await self._async.execute(fn, *args, **kwargs)
        return result


class RetryService:
    """Centralizes sync and async retry policy creation."""

    def __init__(self, config: Any) -> None:
        self._config = config

    def policy_from_config(self) -> Any:
        """Return a RetryPolicy Pydantic schema built from this service's config."""
        from poolgate.schemas.common.ops import RetryPolicy as RetryPolicySchema

        return RetryPolicySchema(
            max_attempts=self._config.max_retries + 1,
            initial_backoff_seconds=self._config.base_backoff,
            max_backoff_seconds=self._config.max_backoff,
            backoff_multiplier=2.0,
        )

    def sync_policy(self, schema: Any = None, max_retries: int | None = None) -> RetryPolicy:
        if schema is not None:
            return RetryPolicy(
                max_retries=schema.max_attempts - 1,
                base_backoff=schema.initial_backoff_seconds,
                max_backoff=schema.max_backoff_seconds,
            )
        return RetryPolicy(
            max_retries=max_retries if max_retries is not None else self._config.max_retries,
            base_backoff=self._config.base_backoff,
            max_backoff=self._config.max_backoff,
        )

    def async_policy(self, schema: Any = None, max_retries: int | None = None) -> AsyncRetryPolicy:
        if schema is not None:
            return AsyncRetryPolicy(
                max_retries=schema.max_attempts - 1,
                base_backoff=schema.initial_backoff_seconds,
                max_backoff=schema.max_backoff_seconds,
            )
        return AsyncRetryPolicy(
            max_retries=max_retries if max_retries is not None else self._config.max_retries,
            base_backoff=self._config.base_backoff,
            max_backoff=self._config.max_backoff,
        )

    def execute(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return self.sync_policy().execute(fn, *args, **kwargs)

    async def async_execute(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return await self.async_policy().execute(fn, *args, **kwargs)
