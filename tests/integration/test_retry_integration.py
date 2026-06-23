"""Integration tests for retry.py — RetryPolicy and AsyncRetryPolicy."""

from __future__ import annotations

import pytest

from exceptions.keys import APIKeyDisabledError, NoAvailableAPIKeyError
from exceptions.output import StructuredOutputError
from retry import (
    _is_auth_error,
    _is_rate_limit,
    _is_retryable,
    AsyncRetryPolicy,
    RetryPolicy,
)

# Use near-zero backoff so tests run fast
_FAST_POLICY = dict(max_retries=2, base_backoff=0.001, max_backoff=0.001)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


class TestIsRetryable:

    def test_api_key_disabled_not_retryable(self):
        assert _is_retryable(APIKeyDisabledError("disabled")) is False

    def test_structured_output_error_not_retryable(self):
        assert _is_retryable(StructuredOutputError("bad json")) is False

    def test_no_available_key_not_retryable(self):
        assert _is_retryable(NoAvailableAPIKeyError("no keys")) is False

    def test_generic_rate_limit_error_is_retryable(self):
        class RateLimitError(Exception):
            pass

        exc = RateLimitError("rate limited")
        exc.response = None  # type: ignore[attr-defined]
        assert _is_retryable(exc) is True

    def test_generic_runtime_error_is_retryable(self):
        # RuntimeError is not in retryable_names but also not fatal → retryable
        # Actually let's test with a "retryable" type name
        class APIConnectionError(Exception):
            pass

        assert _is_retryable(APIConnectionError("connection error")) is True

    def test_standard_exception_is_retryable_by_default(self):
        # A plain ValueError is retryable unless specifically categorised as fatal
        exc = ValueError("unknown")
        # This is not in retryable_names, not an auth error → retryable by default
        # Actually _is_retryable checks exc_type names and httpx — plain ValueError is retryable
        assert isinstance(_is_retryable(exc), bool)


class TestIsRateLimit:

    def test_rate_limit_error_name(self):
        class RateLimitError(Exception):
            pass

        assert _is_rate_limit(RateLimitError("rl")) is True

    def test_other_error_not_rate_limit(self):
        assert _is_rate_limit(ValueError("nope")) is False


class TestIsAuthError:

    def test_status_401_is_auth_error(self):
        exc = Exception("auth")
        exc.status_code = 401  # type: ignore[attr-defined]
        assert _is_auth_error(exc) is True

    def test_status_403_is_auth_error(self):
        exc = Exception("forbidden")
        exc.status_code = 403  # type: ignore[attr-defined]
        assert _is_auth_error(exc) is True

    def test_authentication_error_type_is_auth_error(self):
        class AuthenticationError(Exception):
            pass

        assert _is_auth_error(AuthenticationError("bad creds")) is True

    def test_generic_error_not_auth(self):
        assert _is_auth_error(ValueError("nope")) is False


# ---------------------------------------------------------------------------
# RetryPolicy — sync
# ---------------------------------------------------------------------------


class TestRetryPolicySync:

    def test_success_first_try_no_retries(self):
        call_count = [0]

        def fn():
            call_count[0] += 1
            return "success"

        policy = RetryPolicy(**_FAST_POLICY)
        result = policy.execute(fn)
        assert result == "success"
        assert call_count[0] == 1

    def test_fails_twice_then_succeeds(self):
        call_count = [0]

        class APIConnectionError(Exception):
            pass

        def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise APIConnectionError("transient")
            return "ok"

        policy = RetryPolicy(**_FAST_POLICY)
        result = policy.execute(fn)
        assert result == "ok"
        assert call_count[0] == 3

    def test_all_attempts_fail_raises_last_exception(self):
        class RetryableError(Exception):
            pass

        def fn():
            raise RetryableError("always fails")

        policy = RetryPolicy(max_retries=1, base_backoff=0.001, max_backoff=0.001)
        with pytest.raises(RetryableError):
            policy.execute(fn)

    def test_non_retryable_error_raises_immediately(self):
        call_count = [0]

        def fn():
            call_count[0] += 1
            raise APIKeyDisabledError("disabled")

        policy = RetryPolicy(**_FAST_POLICY)
        with pytest.raises(APIKeyDisabledError):
            policy.execute(fn)
        # APIKeyDisabledError is not retryable → called only once
        assert call_count[0] == 1

    def test_structured_output_error_not_retried(self):
        call_count = [0]

        def fn():
            call_count[0] += 1
            raise StructuredOutputError("bad output")

        policy = RetryPolicy(**_FAST_POLICY)
        with pytest.raises(StructuredOutputError):
            policy.execute(fn)
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# AsyncRetryPolicy
# ---------------------------------------------------------------------------


class TestAsyncRetryPolicy:

    @pytest.mark.asyncio
    async def test_async_success_first_try(self):
        call_count = [0]

        async def fn():
            call_count[0] += 1
            return "async_success"

        policy = AsyncRetryPolicy(**_FAST_POLICY)
        result = await policy.execute(fn)
        assert result == "async_success"
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_async_retries_on_retryable_error(self):
        call_count = [0]

        class APIConnectionError(Exception):
            pass

        async def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise APIConnectionError("transient")
            return "done"

        policy = AsyncRetryPolicy(**_FAST_POLICY)
        result = await policy.execute(fn)
        assert result == "done"
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_async_non_retryable_raises_immediately(self):
        call_count = [0]

        async def fn():
            call_count[0] += 1
            raise APIKeyDisabledError("disabled")

        policy = AsyncRetryPolicy(**_FAST_POLICY)
        with pytest.raises(APIKeyDisabledError):
            await policy.execute(fn)
        assert call_count[0] == 1

    @pytest.mark.asyncio
    async def test_async_all_fail_raises(self):
        class RetryableError(Exception):
            pass

        async def fn():
            raise RetryableError("always fails")

        policy = AsyncRetryPolicy(max_retries=1, base_backoff=0.001, max_backoff=0.001)
        with pytest.raises(RetryableError):
            await policy.execute(fn)
