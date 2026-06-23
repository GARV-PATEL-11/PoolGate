"""Unit tests for retry.py's pure classification functions and policy execution."""

from __future__ import annotations

import httpx
import pytest

from exceptions.keys import APIKeyDisabledError
from exceptions.response import InvalidResponseError
from retry import _is_retryable, ErrorCategory, RetryClassifier, RetryPolicy


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
