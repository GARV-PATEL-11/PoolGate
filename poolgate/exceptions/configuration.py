from __future__ import annotations

from poolgate.exceptions.base import GroqServiceError


class ConfigurationError(GroqServiceError):
    """Raised for missing or invalid configuration at startup."""


class EnvironmentParseError(ConfigurationError):
    def __init__(
        self,
        message: str,
        var_name: str,
        raw_value: str,
        expected: type,
        request_id: str | None = None,
    ) -> None:
        self.var_name = var_name
        self.raw_value = raw_value
        self.expected = expected
        super().__init__(message, request_id)


class InvalidRateLimitConfigError(ConfigurationError):
    def __init__(
        self,
        message: str,
        field: str,
        value: int | float,
        request_id: str | None = None,
    ) -> None:
        self.field = field
        self.value = value
        super().__init__(message, request_id)


class EmptyKeyPoolError(ConfigurationError):
    """Raised when the key list is empty at construction time."""
