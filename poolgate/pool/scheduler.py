"""RequestScheduler — health-aware key selection and dispatch coordination."""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import threading
from typing import TYPE_CHECKING, Any

from poolgate.core.logger import ObservabilityLogger
from poolgate.exceptions.keys import NoAvailableAPIKeyError
from poolgate.pool.key_pool import APIKeyState
from poolgate.pool.strategies.base import BaseSchedulingStrategy, SchedulingStrategyType, create_strategy
from poolgate.schemas.common.runtime import APIKeyStatus

if TYPE_CHECKING:
    from poolgate.core.config import GroqConfig


class RequestScheduler:
    """Thread-safe and async-safe scheduler with pluggable scheduling strategy."""

    def __init__(
        self,
        keys: list[APIKeyState],
        config: GroqConfig,
        logger: ObservabilityLogger,
        strategy: BaseSchedulingStrategy | SchedulingStrategyType | str = SchedulingStrategyType.HEALTH_AWARE,
    ) -> None:
        self._keys = keys
        self._config = config
        self._logger = logger
        self._sync_lock = threading.Lock()
        self._strategy = self._resolve_strategy(strategy)

    @staticmethod
    def _resolve_strategy(
        strategy: BaseSchedulingStrategy | SchedulingStrategyType | str,
    ) -> BaseSchedulingStrategy:
        if isinstance(strategy, BaseSchedulingStrategy):
            return strategy
        return create_strategy(strategy)

    def set_strategy(self, strategy: BaseSchedulingStrategy | SchedulingStrategyType | str) -> None:
        with self._sync_lock:
            self._strategy = self._resolve_strategy(strategy)
            self._logger.debug(f"Scheduler strategy switched to {self._strategy.name()}")

    def current_strategy_name(self) -> str:
        return self._strategy.name()

    def _gather_candidates(self) -> list[APIKeyState]:
        candidates = [
            k for k in self._keys if k.is_available and k.status not in (APIKeyStatus.DISABLED, APIKeyStatus.FAILED)
        ]
        if candidates:
            return candidates
        for k in self._keys:
            if not k.is_cooling_down and k.status == APIKeyStatus.RATE_LIMITED:
                k.reset_cooldown()
        return [
            k for k in self._keys if k.is_available and k.status not in (APIKeyStatus.DISABLED, APIKeyStatus.FAILED)
        ]

    def _select_key(self, request_id: str, *, max_rpm_override: int | None = None) -> APIKeyState:
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
            f"(rpm={best.requests_per_minute}, active={best.active_requests})",
        )
        return best

    def acquire_key(self, request_id: str, *, model: str | None = None) -> APIKeyState:
        max_rpm_override = self._max_rpm_for_model(model) if model else None
        with self._sync_lock:
            key = self._select_key(request_id, max_rpm_override=max_rpm_override)
            key.record_request_start()
            return key

    @staticmethod
    def _max_rpm_for_model(model: str) -> int | None:
        try:
            from poolgate.providers.groq.models import get_model_config

            return get_model_config(model).requests_per_minute
        except Exception:
            return None

    def select_key(self, request_id: str = "") -> APIKeyState:
        return self.acquire_key(request_id)

    @staticmethod
    def release_key(key: APIKeyState, *, latency: float, tokens_in: int, tokens_out: int) -> None:
        key.record_request_end(latency, tokens_in, tokens_out)

    def mark_key_failure(self, key: APIKeyState, *, is_rate_limit: bool = False) -> None:
        key.record_failure(
            is_rate_limit=is_rate_limit,
            cooldown_secs=self._config.cooldown_seconds,
        )
        if is_rate_limit:
            self._logger.warning(
                f"Key {key.masked_key} hit rate limit — cooling down for "
                f"{self._config.cooldown_seconds}s. Consecutive 429s: {key.consecutive_429s}",
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
                    f"{key.consecutive_failures} consecutive non-rate-limit failures.",
                )

    def mark_key_disabled(self, key: APIKeyState) -> None:
        key.mark_disabled()
        self._logger.error(f"Key {key.masked_key} disabled (401/403).")

    async def async_acquire_key(self, request_id: str, *, model: str | None = None) -> APIKeyState:
        loop = asyncio.get_running_loop()
        call = functools.partial(self.acquire_key, request_id, model=model)
        return await loop.run_in_executor(None, call)

    async def aselect_key(self, request_id: str = "") -> APIKeyState:
        return await self.async_acquire_key(request_id)

    def status_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "key_id": k.key_id,
                "masked": k.masked_key,
                "status": k.status.value,
                "requests_per_minute": k.requests_per_minute,
                "rpm": k.requests_per_minute,
                "active": k.active_requests,
                "score": round(k.health_score(self._config.max_rpm_per_key), 2),
                "failure_rate": round(k.failure_rate, 3),
            }
            for k in self._keys
        ]
