from typing import Any

from poolgate.services.retry import (
    AsyncRetryPolicy,
    BackoffCalculator,
    ErrorCategory,
    RetryClassifier,
    RetryExecutor,
    RetryPolicy,
    RetryService,
    is_auth_error,
    is_rate_limit,
)

__all__ = [
    "GroqService",
    "HealthService",
    "RetryPolicy",
    "AsyncRetryPolicy",
    "RetryClassifier",
    "BackoffCalculator",
    "RetryExecutor",
    "RetryService",
    "ErrorCategory",
    "is_auth_error",
    "is_rate_limit",
]


def __getattr__(name: str) -> Any:
    if name == "GroqService":
        from poolgate.services.provider import GroqService

        return GroqService
    if name == "HealthService":
        from poolgate.services.health import HealthService

        return HealthService
    raise AttributeError(f"module 'poolgate.services' has no attribute {name!r}")
