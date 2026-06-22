"""Unit tests for services/retry_service.py."""

from __future__ import annotations

import pytest

from retry import AsyncRetryPolicy, RetryPolicy
from schemas.ops import RetryPolicy as RetryPolicySchema
from services.retry_service import RetryService


@pytest.fixture
def service(monkeypatch) -> RetryService:
	monkeypatch.setenv("TOTAL_GROQ_KEYS", "3")
	monkeypatch.setenv("GROQ_API_KEY_01", "gsk_test_1")
	monkeypatch.setenv("GROQ_API_KEY_02", "gsk_test_2")
	monkeypatch.setenv("GROQ_API_KEY_03", "gsk_test_3")
	from core.config import GroqConfig

	config = GroqConfig.from_env()
	return RetryService(config)


class TestPolicyFromConfig:

	def test_max_attempts_is_retries_plus_one(self, service):
		policy = service.policy_from_config()
		assert policy.max_attempts == service._config.max_retries + 1

	def test_returns_retry_policy_schema(self, service):
		policy = service.policy_from_config()
		assert isinstance(policy, RetryPolicySchema)

	def test_initial_backoff_matches_config(self, service):
		policy = service.policy_from_config()
		assert policy.initial_backoff_seconds == service._config.base_backoff


class TestSyncPolicy:

	def test_returns_retry_policy_instance(self, service):
		policy = service.sync_policy()
		assert isinstance(policy, RetryPolicy)

	def test_custom_policy_overrides_config(self, service):
		custom = RetryPolicySchema(
			max_attempts=2, initial_backoff_seconds=0.1, max_backoff_seconds=1.0,
			)
		policy = service.sync_policy(custom)
		assert isinstance(policy, RetryPolicy)


class TestAsyncPolicy:

	def test_returns_async_retry_policy(self, service):
		policy = service.async_policy()
		assert isinstance(policy, AsyncRetryPolicy)


class TestExecute:

	def test_execute_calls_fn_and_returns_result(self, service):
		result = service.execute(lambda: 42)
		assert result == 42

	def test_execute_propagates_exception_after_exhaustion(self, service):
		def fn():
			raise RuntimeError("permanent")

		with pytest.raises(RuntimeError):
			service.execute(fn)

	@pytest.mark.asyncio
	async def test_async_execute_calls_fn(self, service):
		async def fn():
			return "async_ok"

		result = await service.async_execute(fn)
		assert result == "async_ok"
