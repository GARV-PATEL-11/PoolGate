from __future__ import annotations

from poolgate.exceptions.base import GroqServiceError


class TransportError(GroqServiceError):
    def __init__(
        self,
        message: str,
        attempts: int = 0,
        request_id: str | None = None,
    ) -> None:
        self.attempts = attempts
        super().__init__(message, request_id)


class UpstreamTimeoutError(TransportError):
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
