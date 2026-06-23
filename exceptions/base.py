"""
exceptions/base.py
──────────────────
Root of the PoolGate exception hierarchy.

All public-facing exceptions are subclasses of GroqServiceError, which
guarantees that callers can catch a single type to guard against every
error the library can produce.

Hierarchy
─────────
GroqServiceError
├── ConfigurationError
│   ├── EnvironmentParseError
│   ├── InvalidRateLimitConfigError
│   └── EmptyKeyPoolError
├── InvalidRequestError
│   ├── MissingPromptError
│   ├── InvalidMessageRoleError
│   └── UnknownModelError
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
└── SessionError
    └── SessionExpiredError
"""

from __future__ import annotations


class GroqServiceError(Exception):
    """
    Base exception for all Groq service errors.
    All errors are production-safe — they never expose raw API keys.
    """

    def __init__(self, message: str, request_id: str | None = None) -> None:
        self.request_id = request_id
        super().__init__(message)

    def __repr__(self) -> str:  # pragma: no cover
        cls = type(self).__name__
        rid = f", request_id={self.request_id!r}" if self.request_id else ""
        return f"{cls}({str(self)!r}{rid})"
