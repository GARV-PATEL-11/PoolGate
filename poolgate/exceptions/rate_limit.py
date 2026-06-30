from __future__ import annotations

from poolgate.exceptions.base import GroqServiceError


class RateLimitExceededError(GroqServiceError):
    def __init__(
        self,
        message: str = "Rate limit exceeded across all keys",
        retry_after: float | None = None,
        key_id: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.retry_after = retry_after
        self.key_id = key_id
        super().__init__(message, request_id)


class QuotaExceededError(GroqServiceError):
    def __init__(
        self,
        message: str,
        key_id: str | None = None,
        quota_type: str = "unknown",
        resets_at: str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.key_id = key_id
        self.quota_type = quota_type
        self.resets_at = resets_at
        super().__init__(message, request_id)


class DailyLimitExceededError(QuotaExceededError):
    def __init__(
        self,
        message: str,
        key_id: str | None = None,
        resets_at: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, key_id=key_id, quota_type="daily", resets_at=resets_at, request_id=request_id)


class TokenBudgetExceededError(GroqServiceError):
    def __init__(
        self,
        message: str,
        key_id: str | None = None,
        budget_type: str = "tpm",
        limit: int | None = None,
        retry_after: float | None = None,
        request_id: str | None = None,
    ) -> None:
        self.key_id = key_id
        self.budget_type = budget_type
        self.limit = limit
        self.retry_after = retry_after
        super().__init__(message, request_id)
