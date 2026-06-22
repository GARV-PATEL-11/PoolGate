"""
RequestScheduler — health-aware key selection and dispatch coordination.

Responsibilities:
  - delegate "which key should handle this request" to a pluggable
    SchedulingStrategy (round robin, least used, weighted round robin,
    least remaining capacity, priority failover, or the original
    health-score based selection — see scheduling_strategies.py)
  - enforce per-key requests_per_minute, active-request, and cooldown constraints
  - provide both sync and async key acquisition
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import threading
from typing import TYPE_CHECKING

from exceptions.keys import NoAvailableAPIKeyError
from key_manager.key_pool import APIKeyState
from observability import ObservabilityLogger
from schedulers.scheduling_strategies import (
	BaseSchedulingStrategy,
	SchedulingStrategyType,
	create_strategy,
	)
from schemas.runtime import APIKeyStatus


if TYPE_CHECKING:
	from core.config import GroqConfig


class RequestScheduler:
	"""
	Thread-safe and async-safe scheduler that owns a list of APIKeyState
	objects and exposes acquire / release semantics. The actual "which key
	wins" decision is delegated to a SchedulingStrategy, so the algorithm
	can be swapped without touching dispatch/lifecycle logic.

	Usage (sync):
			key = scheduler.acquire_key(request_id="...")
			try:
					# use key.raw_key
			finally:
					scheduler.release_key(key, latency=..., tokens_in=..., tokens_out=...)

	Usage (async):
			key = await scheduler.async_acquire_key(request_id="...")
			try:
					...
			finally:
					scheduler.release_key(key, ...)

	Choosing a strategy:
			scheduler = RequestScheduler(
					keys, config, logger,
					strategy=SchedulingStrategyType.WEIGHTED_ROUND_ROBIN,
			)
			# or swap at runtime:
			scheduler.set_strategy(SchedulingStrategyType.PRIORITY_FAILOVER)

	Available strategies (see scheduling_strategies.py for details):
			HEALTH_AWARE              - original behavior, scores latency/requests_per_minute/failures
			ROUND_ROBIN               - K1 -> K2 -> K3 -> K1, for equal-capacity keys
			LEAST_USED                - lowest requests_per_minute/active-requests wins, maximizes utilization
			WEIGHTED_ROUND_ROBIN      - higher-capacity keys get proportionally more traffic
			LEAST_REMAINING_CAPACITY  - key with the most unused requests_per_minute budget wins
			PRIORITY_FAILOVER         - primary key until it fails, then next backup
	"""

	def __init__(
			self,
			keys: list[APIKeyState],
			config: GroqConfig,
			logger: ObservabilityLogger,
			strategy: BaseSchedulingStrategy
			          | SchedulingStrategyType
			          | str = SchedulingStrategyType.HEALTH_AWARE,
			) -> None:
		self._keys = keys
		self._config = config
		self._logger = logger
		self._sync_lock = threading.Lock()
		self._strategy = self._resolve_strategy(strategy)

	# ------------------------------------------------------------------
	# Strategy management
	# ------------------------------------------------------------------

	@staticmethod
	def _resolve_strategy(
			strategy: BaseSchedulingStrategy | SchedulingStrategyType | str,
			) -> BaseSchedulingStrategy:
		if isinstance(strategy, BaseSchedulingStrategy):
			return strategy
		return create_strategy(strategy)

	def set_strategy(
			self,
			strategy: BaseSchedulingStrategy | SchedulingStrategyType | str,
			) -> None:
		"""Swap the active scheduling strategy at runtime, thread-safely."""
		with self._sync_lock:
			self._strategy = self._resolve_strategy(strategy)
			self._logger.debug(f"Scheduler strategy switched to {self._strategy.name()}")

	def current_strategy_name(self) -> str:
		return self._strategy.name()

	# ------------------------------------------------------------------
	# Candidate gathering (availability filtering + cooldown revival)
	# ------------------------------------------------------------------

	def _gather_candidates(self) -> list[APIKeyState]:
		candidates = [
			k
			for k in self._keys
			if k.is_available and k.status not in (APIKeyStatus.DISABLED, APIKeyStatus.FAILED)
			]
		if candidates:
			return candidates

		# Second-pass: try to revive keys whose cooldown has lapsed
		for k in self._keys:
			if k.is_cooling_down is False and k.status == APIKeyStatus.RATE_LIMITED:
				k.reset_cooldown()
		return [
			k
			for k in self._keys
			if k.is_available and k.status not in (APIKeyStatus.DISABLED, APIKeyStatus.FAILED)
			]

	def _select_key(self, request_id: str, *, max_rpm_override: int | None = None) -> APIKeyState:
		"""
		Gather available candidates, then hand off the actual pick to the
		active SchedulingStrategy. Raises NoAvailableAPIKeyError if none
		qualify even after attempting cooldown revival.

		max_rpm_override, when given, is used in place of
		self._config.max_rpm_per_key for this selection only — see
		acquire_key()'s docstring for why this exists.
		"""
		candidates = self._gather_candidates()
		if not candidates:
			raise NoAvailableAPIKeyError(
				f"All {len(self._keys)} API keys are unavailable (rate-limited, cooling, or failed).",
				request_id=request_id,
				)

		effective_config = self._config
		if max_rpm_override is not None and max_rpm_override != self._config.max_rpm_per_key:
			effective_config = dataclasses.replace(self._config, max_rpm_per_key=max_rpm_override)

		best = self._strategy.select(candidates, self._keys, effective_config)
		self._logger.debug(
			f"[{self._strategy.name()}] Selected key {best.masked_key} "
			f"(requests_per_minute={best.requests_per_minute}, active={best.active_requests})",
			)
		return best

	def _fallback_select(self, candidates: list[APIKeyState]) -> APIKeyState:
		"""Health-score fallback used by older scheduler callers/tests."""
		return max(candidates, key=lambda key: key.health_score(self._config.max_rpm_per_key))

	# ------------------------------------------------------------------
	# Sync acquire / release
	# ------------------------------------------------------------------

	def acquire_key(self, request_id: str, *, model: str | None = None) -> APIKeyState:
		"""
				model: optional Groq model ID. When given, the per-model requests-
				per-minute limit from llm_models.get_model_config(model) is used
				for this selection's health scoring instead of the pool-wide
				config.max_rpm_per_key default — a whisper-large-v3 call and a
				llama-3.3-70b-versatile call are scored against their own real
				Groq limits rather than one global ceiling. Unknown/unregistered
		model IDs fall back to the pool-wide default silently (model
		validity is enforced separately by clients.assert_capability).
		"""
		max_rpm_override = self._max_rpm_for_model(model) if model else None
		with self._sync_lock:
			key = self._select_key(request_id, max_rpm_override=max_rpm_override)
			key.record_request_start()
			return key

	@staticmethod
	def _max_rpm_for_model(model: str) -> int | None:
		try:
			from llm_models import get_model_config

			return get_model_config(model).requests_per_minute
		except Exception as exc:
			# Unknown model, or a limit type that doesn't apply (e.g. an
			# audio-only model with no requests_per_minute field) — fall
			# back to the pool-wide default rather than failing selection.
			return None

	def select_key(self, request_id: str = "") -> APIKeyState:
		return self.acquire_key(request_id)

	@staticmethod
	def release_key(
			key: APIKeyState,
			*,
			latency: float,
			tokens_in: int,
			tokens_out: int,
			) -> None:
		key.record_request_end(latency, tokens_in, tokens_out)

	def mark_key_failure(
			self,
			key: APIKeyState,
			*,
			is_rate_limit: bool = False,
			) -> None:
		key.record_failure(
			is_rate_limit=is_rate_limit,
			cooldown_secs=self._config.cooldown_seconds,
			)
		if is_rate_limit:
			self._logger.warning(
				f"Key {key.masked_key} hit rate limit — cooling down for "
				f"{self._config.cooldown_seconds}s. "
				f"Consecutive 429s: {key.consecutive_429s}",
				)
		else:
			self._logger.warning(
				f"Key {key.masked_key} recorded a failure. "
				f"Consecutive failures: {key.consecutive_failures}/{self._config.failure_threshold}.",
				)
			if key.consecutive_failures >= self._config.failure_threshold:
				key.mark_failed()
				self._logger.error(
					f"Key {key.masked_key} marked FAILED after "
					f"{key.consecutive_failures} consecutive non-rate-limit failures "
					f"(failure_threshold={self._config.failure_threshold}). "
					"It will not be selected until manually re-enabled via KeyPool.enable_key() "
					"or RequestScheduler equivalent.",
					)

	def mark_key_disabled(self, key: APIKeyState) -> None:
		key.mark_disabled()
		self._logger.error(f"Key {key.masked_key} disabled (401/403).")

	# ------------------------------------------------------------------
	# Async acquire (non-blocking — uses asyncio event loop if needed)
	# ------------------------------------------------------------------

	async def async_acquire_key(self, request_id: str, *, model: str | None = None) -> APIKeyState:
		"""
		Async-safe key acquisition via asyncio.get_running_loop().run_in_executor.
		Selection itself is CPU-bound but very fast, so we run in the default
		executor to avoid blocking the event loop. See acquire_key()'s
		docstring for what `model` does.
		"""
		loop = asyncio.get_running_loop()
		call = functools.partial(self.acquire_key, request_id, model=model)
		return await loop.run_in_executor(None, call)

	async def aselect_key(self, request_id: str = "") -> APIKeyState:
		return await self.async_acquire_key(request_id)

	# ------------------------------------------------------------------
	# Diagnostics
	# ------------------------------------------------------------------

	def status_summary(self) -> list[dict]:
		return [
			{
				"key_id": k.key_id,
				"masked": k.masked_key,
				"status": k.status.value,
				"requests_per_minute": k.requests_per_minute,
				"active": k.active_requests,
				"score": round(k.health_score(self._config.max_rpm_per_key), 2),
				"failure_rate": round(k.failure_rate, 3),
				}
			for k in self._keys
			]
