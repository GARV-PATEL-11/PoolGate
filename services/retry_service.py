"""Retry policy construction and execution helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from core.config import GroqConfig
from retry import AsyncRetryPolicy, RetryPolicy
from schemas.ops import RetryPolicy as RetryPolicySchema


T = TypeVar("T")


class RetryService:
	"""Centralizes sync and async retry policy creation."""

	def __init__(self, config: GroqConfig) -> None:
		self._config = config

	def policy_from_config(self) -> RetryPolicySchema:
		return RetryPolicySchema(
			max_attempts=self._config.max_retries + 1,
			initial_backoff_seconds=self._config.base_backoff,
			max_backoff_seconds=self._config.max_backoff,
			)

	def sync_policy(self, policy: RetryPolicySchema | None = None) -> RetryPolicy:
		p = policy or self.policy_from_config()
		return RetryPolicy(
			max_retries=p.max_attempts - 1,
			base_backoff=p.initial_backoff_seconds,
			max_backoff=p.max_backoff_seconds,
			)

	def async_policy(self, policy: RetryPolicySchema | None = None) -> AsyncRetryPolicy:
		p = policy or self.policy_from_config()
		return AsyncRetryPolicy(
			max_retries=p.max_attempts - 1,
			base_backoff=p.initial_backoff_seconds,
			max_backoff=p.max_backoff_seconds,
			)

	def execute(
			self,
			fn: Callable[..., T],
			*args: Any,
			policy: RetryPolicySchema | None = None,
			**kwargs: Any,
			) -> T:
		return self.sync_policy(policy).execute(fn, *args, **kwargs)

	async def async_execute(
			self,
			fn: Callable[..., Awaitable[T]],
			*args: Any,
			policy: RetryPolicySchema | None = None,
			**kwargs: Any,
			) -> T:
		return await self.async_policy(policy).execute(fn, *args, **kwargs)
