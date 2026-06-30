from __future__ import annotations

from poolgate.exceptions.base import GroqServiceError


class StructuredOutputError(GroqServiceError):
    def __init__(
        self,
        message: str,
        raw_response: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.raw_response = raw_response
        super().__init__(message, request_id)


class SessionError(GroqServiceError):
    """Base class for session-lifecycle exceptions."""


class SessionExpiredError(SessionError):
    def __init__(self, session_id: str, request_id: str | None = None) -> None:
        self.session_id = session_id
        super().__init__(f"Session '{session_id}' has expired", request_id)
