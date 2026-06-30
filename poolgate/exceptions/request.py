from __future__ import annotations

from poolgate.exceptions.base import GroqServiceError


class InvalidRequestError(GroqServiceError):
    """Umbrella exception for caller contract violations."""


class MissingPromptError(InvalidRequestError):
    """Raised when neither prompt= nor messages= is provided."""


class InvalidMessageRoleError(InvalidRequestError):
    def __init__(
        self,
        message: str,
        role: str,
        allowed_roles: set[str],
        request_id: str | None = None,
    ) -> None:
        self.role = role
        self.allowed_roles = frozenset(allowed_roles)
        super().__init__(message, request_id)


class UnknownModelError(InvalidRequestError):
    def __init__(
        self,
        message: str,
        model_id: str,
        available_models: list[str] | None = None,
        request_id: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.available_models = available_models or []
        super().__init__(message, request_id)


class CapabilityError(InvalidRequestError):
    def __init__(
        self,
        message: str,
        model_id: str,
        capability: str,
        supported_capabilities: list[str] | None = None,
        request_id: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.capability = capability
        self.supported_capabilities = supported_capabilities or []
        super().__init__(message, request_id)


class UnknownSchedulingStrategyError(GroqServiceError):
    def __init__(
        self,
        message: str,
        strategy: str,
        available_strategies: list[str] | None = None,
        request_id: str | None = None,
    ) -> None:
        self.strategy = strategy
        self.available_strategies = available_strategies or []
        super().__init__(message, request_id)
