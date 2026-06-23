"""
exceptions/request.py
─────────────────────
Exceptions raised when the caller supplies invalid input.

These are never retryable — the request must be corrected before
re-submission.

All new (taxonomy items marked "add").

InvalidRequestError           — umbrella for caller contract violations
├── MissingPromptError        — neither prompt= nor messages= supplied
│                               replaces bare ValueError at service.py:631, :682
├── InvalidMessageRoleError   — role not in {system, user, assistant, tool}
│                               replaces bare ValueError at schemas/runtime.py
├── CapabilityError           — model lacks requested capability
└── UnknownModelError         — model ID absent from the model registry
                                replaces bare KeyError at models/__init__.py:82

UnknownSchedulingStrategyError — unknown strategy string passed to
                                 set_strategy() / constructor
                                 replaces bare ValueError at
                                 schedulers/scheduling_strategies.py:231
"""

from __future__ import annotations

from exceptions.base import GroqServiceError


class InvalidRequestError(GroqServiceError):
    """
    Umbrella exception for caller contract violations.

    Catch this to handle all input-validation failures with a single
    except clause. Use the more specific subclasses below where possible.

    # No override needed — GroqServiceError.__init__ is sufficient.
    """


class MissingPromptError(InvalidRequestError):
    """
    Raised by service.stream() and service.async_stream() when neither
    prompt= nor messages= is provided.

    # No override needed — InvalidRequestError.__init__ is sufficient.
    """


class InvalidMessageRoleError(InvalidRequestError):
    """
    Raised by ChatMessage.validate_role() when the role is not one of
    {system, user, assistant, tool}.

    Attributes
    ----------
    role          : the invalid role string supplied by the caller
    allowed_roles : frozenset of valid role strings
    """

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
    """
    Raised by models.get_model_config() when the model ID is not in the
    registry.

    Attributes
    ----------
    model_id        : the unrecognised model string
    available_models: sorted list of registered model IDs (for hint messages)

    """

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
    """
    Raised when a model is asked to perform a capability it does not support.

    Attributes
    ----------
    model_id              : requested model
    capability            : requested capability string
    supported_capabilities: capabilities available for that model
    """

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
    """
    Raised by schedulers.scheduling_strategies.create_strategy() when an
    unrecognised strategy string is supplied.


    Kept as a direct GroqServiceError subclass (not under
    InvalidRequestError) because invalid strategy names can also come from
    programmatic configuration, not only direct caller input.

    Attributes
    ----------
    strategy             : the unrecognised strategy string
    available_strategies : sorted list of registered strategy names

    """

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
