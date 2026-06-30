from __future__ import annotations

from typing import TYPE_CHECKING

from poolgate.pool.key_pool import APIKeyState
from poolgate.pool.strategies.base import BaseSchedulingStrategy

if TYPE_CHECKING:
    from poolgate.core.config import GroqConfig


class HealthAwareStrategy(BaseSchedulingStrategy):
    """Picks the key with the highest composite health_score()."""

    def select(
        self,
        candidates: list[APIKeyState],
        all_keys: list[APIKeyState],
        config: GroqConfig,
    ) -> APIKeyState:
        return max(
            candidates,
            key=lambda k: k.health_score(
                max_rpm=config.max_rpm_per_key,
                latency_threshold=config.latency_penalty_threshold,
            ),
        )
