from __future__ import annotations

from poolgate.exceptions.base import GroqServiceError


class APIKeyError(GroqServiceError):
    """Base class for all API-key lifecycle exceptions."""


class NoAvailableAPIKeyError(APIKeyError):
    def __init__(
        self,
        message: str = "No healthy API key available",
        total_keys: int = 0,
        reason_counts: dict[str, int] | None = None,
        request_id: str | None = None,
    ) -> None:
        self.total_keys = total_keys
        self.reason_counts = reason_counts or {}
        super().__init__(message, request_id)


class APIKeyDisabledError(APIKeyError):
    def __init__(
        self,
        key_id: str,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        self.key_id = key_id
        self.status_code = status_code
        super().__init__(f"API key '{key_id}' is disabled or unauthorized", request_id)
