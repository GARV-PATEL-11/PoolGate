"""Unit tests for retry.py's pure classification functions and policy execution."""

from __future__ import annotations

import httpx
import pytest

from exceptions.keys import APIKeyDisabledError, NoAvailableAPIKeyError
from exceptions.output import StructuredOutputError
from exceptions.response import InvalidResponseError
from retry import (
	_is_auth_error,
	_is_rate_limit,
	_is_retryable,
	AsyncRetryPolicy,
	BackoffCalculator,
	ErrorCategory,
	RetryClassifier,
	RetryExecutor,
	RetryPolicy,
)


class TestIsRetryable:

	def test_invalid_response_with_retryable_status_is_retryable(self):
		exc = InvalidResponseError("server error", status_code=503)
		assert _is_retryable(exc) is True

	def test_invalid_response_with_fatal_status_is_not_retryable(self):
		exc = InvalidResponseError("bad request", status_code=400)
		assert _is_retryable(exc) is False

	def test_api_key_disabled_is_never_retryable(self):
		exc = APIKeyDisabledError(key_id="key_0", status_code=401)
		assert _is_retryable(exc) is False

	def test_httpx_connect_error_is_retryable(self):
		assert _is_retryable(httpx.ConnectError("boom")) is True

	def test_generic_exception_is_not_retryable(self):
		assert _is_retryable(ValueError("not a provider error")) is False


class TestRetryClassifier:

	def test_classifies_quota_errors(self):
		classifier = RetryClassifier()
		exc = APIKeyDisabledError(key_id="key_0")
		assert classifier.classify(exc) == ErrorCategory.QUOTA

	def test_classifies_provider_errors(self):
		classifier = RetryClassifier()
		exc = InvalidResponseError("server error", status_code=502)
		assert classifier.classify(exc) == ErrorCategory.PROVIDER

	def test_classifies_validation_errors(self):
		classifier = RetryClassifier()
		exc = InvalidResponseError("bad request", status_code=400)
		assert classifier.classify(exc) == ErrorCategory.VALIDATION


class TestRetryPolicyExecution:
	"""
	These exercise the real tenacity-backed retry loop, not just the
	classification helpers — confirmed working against a real (non-stub)
	tenacity installation in CI.
	"""

	def test_retries_until_success_within_budget(self):
		policy = RetryPolicy(max_retries=3, base_backoff=0.01, max_backoff=0.05)
		calls = {"n": 0}

		def flaky():
			calls["n"] += 1
			if calls["n"] < 3:
				raise httpx.ConnectError("transient")
			return "ok"

		assert policy.execute(flaky) == "ok"
		assert calls["n"] == 3

	def test_does_not_retry_non_retryable_errors(self):
		policy = RetryPolicy(max_retries=3, base_backoff=0.01, max_backoff=0.05)
		calls = {"n": 0}

		def always_fails():
			calls["n"] += 1
			raise APIKeyDisabledError(key_id="key_0", status_code=401)

		with pytest.raises(APIKeyDisabledError):
			policy.execute(always_fails)
		assert calls["n"] == 1

	def test_exhausts_retry_budget_and_raises_last_error(self):
		policy = RetryPolicy(max_retries=2, base_backoff=0.01, max_backoff=0.05)
		calls = {"n": 0}

		def always_transient():
			calls["n"] += 1
			raise httpx.ConnectError("persistent failure")

		with pytest.raises(httpx.ConnectError):
			policy.execute(always_transient)
		assert calls["n"] == 3  # initial attempt + 2 retries


class TestIsRetryableAdditional:

	def test_no_available_api_key_is_not_retryable(self):
		exc = NoAvailableAPIKeyError()
		assert _is_retryable(exc) is False

	def test_structured_output_error_is_not_retryable(self):
		exc = StructuredOutputError("bad json")
		assert _is_retryable(exc) is False

	def test_invalid_response_unknown_status_is_not_retryable(self):
		exc = InvalidResponseError("weird error", status_code=418)
		assert _is_retryable(exc) is False

	def test_timeout_exception_is_retryable(self):
		assert _is_retryable(httpx.TimeoutException("timeout")) is True

	def test_remote_protocol_error_is_retryable(self):
		assert _is_retryable(httpx.RemoteProtocolError("proto err")) is True


class TestIsRateLimit:

	def test_rate_limit_invalid_response_429(self):
		exc = InvalidResponseError("rate limit", status_code=429)
		assert _is_rate_limit(exc) is True

	def test_non_429_invalid_response_is_not_rate_limit(self):
		exc = InvalidResponseError("server err", status_code=500)
		assert _is_rate_limit(exc) is False

	def test_generic_exception_is_not_rate_limit(self):
		assert _is_rate_limit(ValueError("nope")) is False


class TestIsAuthError:

	def test_401_is_auth_error(self):
		exc = InvalidResponseError("unauthorized", status_code=401)
		assert _is_auth_error(exc) is True

	def test_403_is_auth_error(self):
		exc = InvalidResponseError("forbidden", status_code=403)
		assert _is_auth_error(exc) is True

	def test_500_is_not_auth_error(self):
		exc = InvalidResponseError("server err", status_code=500)
		assert _is_auth_error(exc) is False

	def test_generic_exception_is_not_auth_error(self):
		assert _is_auth_error(ValueError("nope")) is False


class TestBackoffCalculator:

	def test_first_attempt_returns_base_backoff(self):
		calc = BackoffCalculator(base_backoff=1.0, max_backoff=30.0, jitter=0)
		result = calc.calculate(1)
		assert result == 1.0

	def test_second_attempt_doubles_base(self):
		calc = BackoffCalculator(base_backoff=1.0, max_backoff=30.0, jitter=0)
		result = calc.calculate(2)
		assert result == 2.0

	def test_delay_is_capped_at_max_backoff(self):
		calc = BackoffCalculator(base_backoff=1.0, max_backoff=5.0, jitter=0)
		result = calc.calculate(100)
		assert result == 5.0

	def test_jitter_adds_randomness(self):
		calc = BackoffCalculator(base_backoff=1.0, max_backoff=30.0, jitter=0.5)
		result = calc.calculate(1)
		assert 1.0 <= result <= 1.5

	def test_zero_jitter_is_deterministic(self):
		calc = BackoffCalculator(base_backoff=2.0, max_backoff=10.0, jitter=0)
		assert calc.calculate(1) == 2.0
		assert calc.calculate(1) == 2.0


class TestRetryClassifierAdditional:

	def test_no_available_key_classifies_as_quota(self):
		classifier = RetryClassifier()
		assert classifier.classify(NoAvailableAPIKeyError()) == ErrorCategory.QUOTA

	def test_structured_output_error_classifies_as_validation(self):
		classifier = RetryClassifier()
		assert classifier.classify(StructuredOutputError("bad")) == ErrorCategory.VALIDATION

	def test_rate_limit_invalid_response_classifies_as_quota(self):
		classifier = RetryClassifier()
		exc = InvalidResponseError("too many", status_code=429)
		assert classifier.classify(exc) == ErrorCategory.QUOTA

	def test_unknown_status_classifies_as_unknown(self):
		classifier = RetryClassifier()
		exc = InvalidResponseError("weird", status_code=418)
		assert classifier.classify(exc) == ErrorCategory.UNKNOWN

	def test_is_retriable_delegates_to_is_retryable(self):
		classifier = RetryClassifier()
		assert classifier.is_retriable(httpx.ConnectError("boom")) is True
		assert classifier.is_retriable(APIKeyDisabledError(key_id="k")) is False


class TestRetryExecutor:

	def test_sync_execute_returns_result_on_success(self):
		executor = RetryExecutor(max_retries=1, base_backoff=0.01, max_backoff=0.05)
		assert executor.execute(lambda: "done") == "done"

	def test_sync_execute_retries_on_transient_error(self):
		executor = RetryExecutor(max_retries=2, base_backoff=0.01, max_backoff=0.05)
		calls = {"n": 0}

		def flaky():
			calls["n"] += 1
			if calls["n"] < 2:
				raise httpx.ConnectError("transient")
			return "ok"

		assert executor.execute(flaky) == "ok"
		assert calls["n"] == 2

	@pytest.mark.asyncio
	async def test_aexecute_returns_result_on_success(self):
		executor = RetryExecutor(max_retries=1, base_backoff=0.01, max_backoff=0.05)

		async def fn():
			return "async done"

		result = await executor.aexecute(fn)
		assert result == "async done"

	@pytest.mark.asyncio
	async def test_aexecute_retries_on_transient_error(self):
		executor = RetryExecutor(max_retries=2, base_backoff=0.01, max_backoff=0.05)
		calls = {"n": 0}

		async def flaky():
			calls["n"] += 1
			if calls["n"] < 2:
				raise httpx.ConnectError("transient")
			return "async ok"

		result = await executor.aexecute(flaky)
		assert result == "async ok"
		assert calls["n"] == 2


class TestAsyncRetryPolicy:

	@pytest.mark.asyncio
	async def test_executes_successfully_on_first_attempt(self):
		policy = AsyncRetryPolicy(max_retries=1, base_backoff=0.01, max_backoff=0.05)

		async def fn():
			return "success"

		result = await policy.execute(fn)
		assert result == "success"

	@pytest.mark.asyncio
	async def test_retries_async_transient_errors(self):
		policy = AsyncRetryPolicy(max_retries=3, base_backoff=0.01, max_backoff=0.05)
		calls = {"n": 0}

		async def flaky():
			calls["n"] += 1
			if calls["n"] < 2:
				raise httpx.ConnectError("transient")
			return "ok"

		result = await policy.execute(flaky)
		assert result == "ok"

	@pytest.mark.asyncio
	async def test_does_not_retry_non_retryable_async_errors(self):
		policy = AsyncRetryPolicy(max_retries=3, base_backoff=0.01, max_backoff=0.05)
		calls = {"n": 0}

		async def always_fails():
			calls["n"] += 1
			raise APIKeyDisabledError(key_id="k", status_code=401)

		with pytest.raises(APIKeyDisabledError):
			await policy.execute(always_fails)
		assert calls["n"] == 1
