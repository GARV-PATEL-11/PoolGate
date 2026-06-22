"""
exceptions/transport.py
───────────────────────
Exceptions for network-level failures and upstream (Groq-side) 5xx errors.

Previously, retry.py classified httpx.ConnectError, httpx.TimeoutException,
APIConnectionError, APITimeoutError, InternalServerError, and
ServiceUnavailableError as retryable — but after retry exhaustion it did
`raise last_exc` directly, leaking raw third-party types to callers and
breaking the contract that all public errors are GroqServiceError subclasses.
The provider facade now wraps exhausted retries with RetryExhaustedError;
these transport classes remain available for callers that want finer-grained
mapping at HTTP/API boundaries.

These exceptions normalise those paths.

All new (taxonomy items marked "add").

TransportError           — network / connectivity failures (base)
└── UpstreamTimeoutError — httpx / SDK timeout after retries exhausted

UpstreamServiceError     — Groq 5xx after retries exhausted
"""

from __future__ import annotations

from exceptions.base import GroqServiceError


class TransportError(GroqServiceError):
    """
    Raised when a request fails due to a network-layer error after all
    retry attempts are exhausted.

    Wraps the original httpx or Groq SDK exception as __cause__ so callers
    who need the raw exception can still access it via exc.__cause__.

    Sources classified as retryable in retry.py:
    - httpx.ConnectError
    - httpx.RemoteProtocolError
    - groq.APIConnectionError

    Attributes
    ----------
    attempts : number of attempts made before giving up

    Usage
    -----
    ::

        # service.py — _run_with_rotation(), after retry loop
        except (httpx.ConnectError, APIConnectionError) as exc:
            raise TransportError(
                f"Connection failed after {attempts} attempt(s): {exc}",
                attempts=attempts,
                request_id=request_id,
            ) from exc
    """

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        request_id: str | None = None,
    ) -> None:
        self.attempts = attempts
        super().__init__(message, request_id)


class UpstreamTimeoutError(TransportError):
    """
    Raised when a request times out after all retry attempts are exhausted.

    Subclasses TransportError so callers can catch the general case or the
    timeout-specific case depending on their needs.

    Sources classified as retryable in retry.py:
    - httpx.TimeoutException
    - groq.APITimeoutError

    Attributes
    ----------
    attempts        : number of attempts made before giving up
    timeout_seconds : the configured timeout that was exceeded, if known

    Usage
    -----
    ::

        # service.py — _run_with_rotation(), after retry loop
        except (httpx.TimeoutException, APITimeoutError) as exc:
            raise UpstreamTimeoutError(
                f"Request timed out after {attempts} attempt(s).",
                attempts=attempts,
                timeout_seconds=self._config.timeout,
                request_id=request_id,
            ) from exc
    """

    def __init__(
        self,
        message: str,
        attempts: int = 0,
        timeout_seconds: float | None = None,
        request_id: str | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(message, attempts, request_id)


class UpstreamServiceError(GroqServiceError):
    """
    Raised when Groq returns a 5xx status after all retry attempts are
    exhausted.

    Sources classified as retryable in retry.py:59-72:
    - groq.InternalServerError (500)
    - groq.ServiceUnavailableError (503)
    - Any groq.APIStatusError with a 5xx status code

    Attributes
    ----------
    status_code : HTTP status code returned (500, 503, …)
    attempts    : number of attempts made before giving up

    Usage
    -----
    ::

        # service.py — _run_with_rotation(), after retry loop
        except (InternalServerError, ServiceUnavailableError) as exc:
            raise UpstreamServiceError(
                f"Groq returned {exc.status_code} after {attempts} attempt(s).",
                status_code=exc.status_code,
                attempts=attempts,
                request_id=request_id,
            ) from exc
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        attempts: int = 0,
        request_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.attempts = attempts
        super().__init__(message, request_id)
