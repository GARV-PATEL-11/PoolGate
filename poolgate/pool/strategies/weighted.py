from __future__ import annotations

from typing import TYPE_CHECKING

from poolgate.pool.key_pool import APIKeyState
from poolgate.pool.strategies.base import BaseSchedulingStrategy

if TYPE_CHECKING:
    from poolgate.core.config import GroqConfig


class WeightedRoundRobinStrategy(BaseSchedulingStrategy):
    """Smooth weighted round robin — higher-capacity keys get proportionally more traffic."""

    def __init__(self) -> None:
        self._current_weight: dict[str, float] = {}

    @staticmethod
    def _weight_of(key: APIKeyState, config: GroqConfig) -> float:
        explicit = getattr(key, "weight", None)
        if explicit is not None:
            return float(explicit)
        per_key_limit = getattr(key, "max_rpm", None)
        if per_key_limit is not None:
            return float(per_key_limit)
        return float(config.max_rpm_per_key or 1)

    def select(
        self,
        candidates: list[APIKeyState],
        all_keys: list[APIKeyState],
        config: GroqConfig,
    ) -> APIKeyState:
        total_weight = sum(self._weight_of(k, config) for k in candidates)
        if total_weight <= 0:
            return candidates[0]
        best: APIKeyState | None = None
        best_weight = float("-inf")
        for k in candidates:
            updated = self._current_weight.get(k.key_id, 0.0) + self._weight_of(k, config)
            self._current_weight[k.key_id] = updated
            if updated > best_weight:
                best_weight = updated
                best = k
        assert best is not None
        self._current_weight[best.key_id] -= total_weight
        return best
