from __future__ import annotations

from poolgate.exceptions.base import GroqServiceError


class InvalidResponseError(GroqServiceError):
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
