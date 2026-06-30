from __future__ import annotations

from typing import TYPE_CHECKING

from poolgate.pool.key_pool import APIKeyState
from poolgate.pool.strategies.base import BaseSchedulingStrategy

if TYPE_CHECKING:
    from poolgate.core.config import GroqConfig


class LeastUsedStrategy(BaseSchedulingStrategy):
    """Picks the key with the lowest current-window request count."""

    def select(
        self,
        candidates: list[APIKeyState],
        all_keys: list[APIKeyState],
        config: GroqConfig,
    ) -> APIKeyState:
        return min(candidates, key=lambda k: (k.requests_per_minute, k.active_requests))


class LeastRemainingCapacityStrategy(BaseSchedulingStrategy):
    """Picks the key with the most remaining RPM budget left this window."""

    @staticmethod
    def _limit_of(key: APIKeyState, config: GroqConfig) -> float:
        return float(getattr(key, "max_rpm", None) or config.max_rpm_per_key)

    def select(
        self,
        candidates: list[APIKeyState],
        all_keys: list[APIKeyState],
        config: GroqConfig,
    ) -> APIKeyState:
        return max(candidates, key=lambda k: self._limit_of(k, config) - k.requests_per_minute)
