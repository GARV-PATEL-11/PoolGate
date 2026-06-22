"""
exceptions/rate_limit.py
────────────────────────
Exceptions for 429-family responses from Groq.

Groq returns HTTP 429 for three distinct situations that need different
handling strategies. This module gives each its own typed exception so
callers and the retry machinery can distinguish them without parsing
response bodies.

┌──────────────────────────┬───────────────────┬────────────────────────┐
│ Exception                │ Groq scenario     │ Recovery window        │
├──────────────────────────┼───────────────────┼────────────────────────┤
│ RateLimitExceededError   │ per-minute RPM    │ retry_after seconds    │
│ TokenBudgetExceededError │ TPM / ITPM / OTPM │ reset at minute window │
│ QuotaExceededError       │ daily limit hit   │ resets at midnight UTC │
│   DailyLimitExceededError│ explicit daily    │ same                   │
└──────────────────────────┴───────────────────┴────────────────────────┘

RateLimitExceededError — kept fully compatible, retry_after preserved,
                         default message preserved

QuotaExceededError      — daily / total quota exhaustion (long-lived 429)
DailyLimitExceededError — explicit daily-limit breach (subclass)
TokenBudgetExceededError— TPM / ITPM / OTPM limit breach
"""

from __future__ import annotations

from exceptions.base import GroqServiceError


class RateLimitExceededError(GroqServiceError):
    """
    Raised when a key hits Groq's per-minute request-rate limit (HTTP 429,
    RPM-class).

    The retry_after value is parsed from the Retry-After response header
    when present. The retry machinery uses it to schedule the next attempt
    and to mark the key as cooling.

    Attributes
    ----------
    retry_after : seconds to wait before the key is usable again
                              (None if the header was absent)
    key_id      : the key that was rate-limited
    """

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
    """
    Raised when a key has exhausted a multi-hour or daily quota (HTTP 429,
    quota-class).

    Unlike RateLimitExceededError, a quota breach does not recover within
    seconds — the window is hours. The retry layer must not re-queue the
    same key; it should mark it EXHAUSTED and rotate to a different key,
    or surface this error to the user.

    This also closes the APIKeyStatus.EXHAUSTED dead-code issue: setting
    that status should happen when this exception is caught by the key pool.

    Attributes
    ----------
    key_id     : the key whose quota is exhausted
    quota_type : which quota was hit ("daily", "monthly", etc.)
    resets_at  : ISO-8601 timestamp when the quota resets, if known
    """

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
    """
    Convenience subclass for explicit daily-limit breaches.

    Raised when the response body clearly identifies the limit as a daily
    cap (as opposed to a monthly or total quota). Callers that need to
    treat daily exhaustion differently from other quota types can catch
    this more specific exception.

    """

    def __init__(
        self,
        message: str,
        key_id: str | None = None,
        resets_at: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(
            message,
            key_id=key_id,
            quota_type="daily",
            resets_at=resets_at,
            request_id=request_id,
        )


class TokenBudgetExceededError(GroqServiceError):
    """
    Raised when a request exceeds a token-throughput limit (TPM, ITPM, or
    OTPM) — returned as HTTP 429 by Groq but distinct from RPM and quota
    exhaustion.

    The model registry already tracks per-model TPM limits
    (ModelRateLimitConfig), but nothing currently raises a typed error when
    those limits are breached. This exception closes that gap.

    Attributes
    ----------
    key_id      : the key that received the 429
    budget_type : which token budget was hit: "tpm", "itpm", or "otpm"
    limit       : the limit value (tokens/minute), if parseable from the response
    retry_after : seconds until the budget resets (typically < 60)

    """

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
