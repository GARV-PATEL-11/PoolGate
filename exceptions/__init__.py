"""
exceptions/__init__.py
───────────────────────
PoolGate exception hierarchy — public surface.

All exceptions are importable directly from this package::

    from exceptions import (
        GroqServiceError,
        ConfigurationError,
        EnvironmentParseError,
        InvalidRateLimitConfigError,
        EmptyKeyPoolError,
        InvalidRequestError,
        MissingPromptError,
        InvalidMessageRoleError,
        UnknownModelError,
        UnknownSchedulingStrategyError,
        APIKeyError,
        NoAvailableAPIKeyError,
        APIKeyDisabledError,
        RateLimitExceededError,
        QuotaExceededError,
        DailyLimitExceededError,
        TokenBudgetExceededError,
        TransportError,
        UpstreamTimeoutError,
        UpstreamServiceError,
        InvalidResponseError,
        RetryExhaustedError,
        StructuredOutputError,
        SessionError,
        SessionExpiredError,
    )

Full hierarchy
──────────────
GroqServiceError
├── ConfigurationError
│   ├── EnvironmentParseError
│   ├── InvalidRateLimitConfigError
│   └── EmptyKeyPoolError
├── InvalidRequestError
│   ├── MissingPromptError
│   ├── InvalidMessageRoleError
│   ├── UnknownModelError
│   └── CapabilityError
├── UnknownSchedulingStrategyError
├── APIKeyError
│   ├── NoAvailableAPIKeyError
│   └── APIKeyDisabledError
├── RateLimitExceededError
├── QuotaExceededError
│   └── DailyLimitExceededError
├── TokenBudgetExceededError
├── TransportError
│   └── UpstreamTimeoutError
├── UpstreamServiceError
├── InvalidResponseError
├── RetryExhaustedError
├── StructuredOutputError
├── SessionError
│   └── SessionExpiredError
└── ThrottleError
    ├── CapabilityThrottledError
    └── ModelThrottledError
"""

from exceptions.base import GroqServiceError
from exceptions.configuration import (
    ConfigurationError,
    EmptyKeyPoolError,
    EnvironmentParseError,
    InvalidRateLimitConfigError,
)
from exceptions.keys import (
    APIKeyDisabledError,
    APIKeyError,
    NoAvailableAPIKeyError,
)
from exceptions.output import (
    SessionError,
    SessionExpiredError,
    StructuredOutputError,
)
from exceptions.persistence import PersistenceError
from exceptions.rate_limit import (
    DailyLimitExceededError,
    QuotaExceededError,
    RateLimitExceededError,
    TokenBudgetExceededError,
)
from exceptions.request import (
    CapabilityError,
    InvalidMessageRoleError,
    InvalidRequestError,
    MissingPromptError,
    UnknownModelError,
    UnknownSchedulingStrategyError,
)
from exceptions.response import (
    InvalidResponseError,
    RetryExhaustedError,
)
from exceptions.throttle import (
    CapabilityThrottledError,
    ModelThrottledError,
    ThrottleError,
)
from exceptions.transport import (
    TransportError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)

__all__ = [
    # Root
    "GroqServiceError",
    # Configuration (startup)
    "ConfigurationError",
    "EnvironmentParseError",
    "InvalidRateLimitConfigError",
    "EmptyKeyPoolError",
    # Request / input validation
    "InvalidRequestError",
    "MissingPromptError",
    "InvalidMessageRoleError",
    "UnknownModelError",
    "UnknownSchedulingStrategyError",
    "CapabilityError",
    # Key lifecycle
    "APIKeyError",
    "NoAvailableAPIKeyError",
    "APIKeyDisabledError",
    # Rate limiting (429-family)
    "RateLimitExceededError",
    "QuotaExceededError",
    "DailyLimitExceededError",
    "TokenBudgetExceededError",
    # Transport / network
    "TransportError",
    "UpstreamTimeoutError",
    "UpstreamServiceError",
    # Response parsing
    "InvalidResponseError",
    "RetryExhaustedError",
    # Output / session
    "StructuredOutputError",
    "SessionError",
    "SessionExpiredError",
    "PersistenceError",
    # Throttling
    "ThrottleError",
    "CapabilityThrottledError",
    "ModelThrottledError",
]
