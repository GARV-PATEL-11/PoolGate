"""
exceptions/response.py
───────────────────────
Exceptions for malformed API responses and retry exhaustion.

Existing (from your exceptions.py — both had issues)
------------------------------------------------------
InvalidResponseError — was defined and referenced in retry._is_retryable()
                       but NEVER raised anywhere. Four unguarded choices[0]
                       accesses in sync + async clients produced bare
                       IndexError / AttributeError instead.
                       Status: defined here with matching signature and used
                       by client response parsing helpers.

New (taxonomy items marked "add")
----------------------------------
RetryExhaustedError — _run_with_rotation() did `raise last_exc` directly,
                      leaking raw httpx / SDK types to callers.
                      Now raises RetryExhaustedError(...) from last_exc.
"""

from __future__ import annotations

from exceptions.base import GroqServiceError


class InvalidResponseError(GroqServiceError):
    """
    Raised when the Groq API returns a structurally unexpected payload —
    e.g. an empty choices list, a missing delta field in a stream chunk,
    or a response that cannot be normalised into a GroqResponse.

    Previously broken
    -----------------
    Defined in exceptions.py and referenced in retry._is_retryable(), but
    no code path ever constructed and raised it. The four choices[0]
    accesses in sync_client.py:105, async_client.py:105 (completion) and
    sync_client.py:148, async_client.py:148 (streaming chunk) were
    completely unguarded — a content-filtered response caused bare
    IndexError.

    Client usage
    ------------
    ::

            # clients/sync_client.py:105 (and async equivalent)
            try:
                    choice = completion.choices[0]
            except IndexError:
                    raise InvalidResponseError(
                            "Groq returned a completion with an empty choices list. "
                            "The request may have been content-filtered.",
                            status_code=200,
                            request_id=request_id,
                    )

            # clients/sync_client.py:148 (streaming chunk, and async equivalent)
            try:
                    delta = chunk.choices[0].delta
            except (IndexError, AttributeError):
                    raise InvalidResponseError(
                            f"Malformed stream chunk — choices[0] unavailable: {chunk!r}",
                            request_id=request_id,
                    )

    Attributes
    ----------
    status_code  : HTTP status of the response, when applicable
    raw_response : the raw object that triggered the error (omit large
                               streaming payloads)
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        raw_response: object | None = None,
        request_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.raw_response = raw_response
        super().__init__(message, request_id)


class RetryExhaustedError(GroqServiceError):
    """
    Raised when _run_with_rotation() / _async_run_with_rotation() exhaust
    all retry attempts without a successful response.

    Previously broken
    -----------------
    Both methods did `raise last_exc` directly after the retry loop,
    leaking raw httpx / Groq SDK types and breaking the contract that all
    public errors are GroqServiceError subclasses.

    Service usage
    -------------
    ::

            # service.py — _run_with_rotation() and _async_run_with_rotation()
            # Replace:  raise last_exc
            # With:
            raise RetryExhaustedError(
                    f"All {attempts} attempt(s) failed. Last error: {last_exc}",
                    attempts=attempts,
                    last_exc=last_exc,
                    request_id=request_id,
            ) from last_exc

    Note on the GroqServiceError fallback
    --------------------------------------
    service.py:243 and :341 also raise GroqServiceError directly as a
    fallback when last_exc is None — but in practice last_exc is almost
    never None because the loop only exits after catching an exception.
    RetryExhaustedError replaces both paths.

    Attributes
    ----------
    attempts : total number of attempts made (initial + retries)
    last_exc : the final exception that caused the last attempt to fail;
                       also available as exc.__cause__ after `raise ... from last_exc`
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exc: BaseException | None = None,
        request_id: str | None = None,
    ) -> None:
        self.attempts = attempts
        self.last_exc = last_exc
        super().__init__(message, request_id)
