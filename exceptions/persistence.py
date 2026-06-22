"""Exceptions for persistence backend failures."""

from __future__ import annotations

from exceptions.base import GroqServiceError


class PersistenceError(GroqServiceError):
    """Raised when a persistence backend cannot load or save tracker data."""

    def __init__(
        self,
        message: str,
        backend: str | None = None,
        path: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.backend = backend
        self.path = path
        super().__init__(message, request_id)
