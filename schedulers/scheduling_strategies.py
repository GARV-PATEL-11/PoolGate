"""
scheduling_strategies.py — pluggable key-selection algorithms for RequestScheduler.

Each strategy implements `select(candidates, all_keys, config) -> APIKeyState`.

Strategies are always invoked while the scheduler holds its `_sync_lock`
(see request_scheduler.py), so any internal state a strategy keeps
(round-robin cursors, weighted-round-robin credits, etc.) does not need
its own locking — selection calls are already serialized by the caller.

Some strategies (Weighted Round Robin, Least Remaining Capacity, Priority
Failover) work best if APIKeyState exposes a few optional per-key
attributes:
  - `weight`    (float)  relative traffic share, used by Weighted Round Robin
  - `max_rpm`   (float)  this key's own rate limit, used by WRR / LRC
  - `priority`  (int)    lower = higher priority, used by Priority Failover
If those attributes are absent, the strategies fall back to sensible
defaults (config.max_rpm_per_key, or the key's position in the pool) so
everything still works against the current APIKeyState implementation
without modification. Add the attributes to key_pool.py later for finer
control.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, TYPE_CHECKING

from exceptions.request import UnknownSchedulingStrategyError
from key_manager.key_pool import APIKeyState

if TYPE_CHECKING:
    from core.config import GroqConfig


# ----------------------------------------------------------------------
# Base interface
# ----------------------------------------------------------------------


class BaseSchedulingStrategy(ABC):
    """Common interface every scheduling strategy must implement."""

    @abstractmethod
    def select(
        self,
        candidates: list[APIKeyState],
        all_keys: list[APIKeyState],
        config: GroqConfig,
    ) -> APIKeyState:
        """
        Pick one key out of `candidates` (already filtered for availability).
        `all_keys` is the full, stably-ordered pool — useful for strategies
        that need positional/priority context beyond just the candidates
        (round robin's cursor, priority failover's primary-key ordering).
        """
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__


SchedulingStrategy = BaseSchedulingStrategy


# ----------------------------------------------------------------------
# 1. Health-Aware — original behavior, best when key health is uncertain
# ----------------------------------------------------------------------


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


# ----------------------------------------------------------------------
# 2. Round Robin — best for a pool of equal-capacity keys
# ----------------------------------------------------------------------


class RoundRobinStrategy(BaseSchedulingStrategy):
    """Rotates through keys sequentially: K1 -> K2 -> K3 -> K1 ..."""

    def __init__(self) -> None:
        self._next_index = 0

    def select(
        self,
        candidates: list[APIKeyState],
        all_keys: list[APIKeyState],
        config: GroqConfig,
    ) -> APIKeyState:
        candidate_ids = {c.key_id for c in candidates}
        n = len(all_keys)
        for offset in range(n):
            idx = (self._next_index + offset) % n
            key = all_keys[idx]
            if key.key_id in candidate_ids:
                self._next_index = (idx + 1) % n
                return key
        # Unreachable in practice since candidates is guaranteed non-empty,
        # but keep a safe fallback rather than raising.
        return candidates[0]


# ----------------------------------------------------------------------
# 3. Least Used — best for maximizing utilization across the pool
# ----------------------------------------------------------------------


class LeastUsedStrategy(BaseSchedulingStrategy):
    """Picks the key with the lowest current-window request count."""

    def select(
        self,
        candidates: list[APIKeyState],
        all_keys: list[APIKeyState],
        config: GroqConfig,
    ) -> APIKeyState:
        return min(
            candidates,
            key=lambda k: (k.requests_per_minute, k.active_requests),
        )


# ----------------------------------------------------------------------
# 4. Weighted Round Robin — best for keys with different capacities
# ----------------------------------------------------------------------


class WeightedRoundRobinStrategy(BaseSchedulingStrategy):
    """
    Smooth weighted round robin (the "current weight" algorithm Nginx
    uses for upstreams). A 10 requests_per_minute key with weight=10 gets ~2x the traffic
    of a 5 requests_per_minute key with weight=5, while still interleaving requests
    rather than draining one key before touching the next.
    """

    def __init__(self) -> None:
        self._current_weight: dict[str, float] = {}

    @staticmethod
    def _weight_of(key: APIKeyState, config: GroqConfig) -> float | int:
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


# ----------------------------------------------------------------------
# 5. Least Remaining Capacity — best for rate-limit-aware scheduling
# ----------------------------------------------------------------------


class LeastRemainingCapacityStrategy(BaseSchedulingStrategy):
    """Picks the key with the most remaining requests_per_minute budget left this window."""

    @staticmethod
    def _limit_of(key: APIKeyState, config: GroqConfig) -> float:
        return float(getattr(key, "max_rpm", None) or config.max_rpm_per_key)

    def select(
        self,
        candidates: list[APIKeyState],
        all_keys: list[APIKeyState],
        config: GroqConfig,
    ) -> APIKeyState:
        def remaining(k: APIKeyState) -> float:
            return self._limit_of(k, config) - k.requests_per_minute

        return max(candidates, key=remaining)


# ----------------------------------------------------------------------
# 6. Priority Failover — best for reliability (primary + backups)
# ----------------------------------------------------------------------


class PriorityFailoverStrategy(BaseSchedulingStrategy):
    """
    Always prefers the primary key; only falls back to a backup once the
    primary is rate-limited, failed, or otherwise unavailable. Priority
    comes from an optional `key.priority` attribute (lower = primary),
    falling back to the key's position in the configured key list.
    """

    def select(
        self,
        candidates: list[APIKeyState],
        all_keys: list[APIKeyState],
        config: GroqConfig,
    ) -> APIKeyState:
        position = {k.key_id: i for i, k in enumerate(all_keys)}

        def priority_of(k: APIKeyState) -> Any:
            explicit = getattr(k, "priority", None)
            return explicit if explicit is not None else position[k.key_id]

        return min(candidates, key=priority_of)


# ----------------------------------------------------------------------
# Registry / factory
# ----------------------------------------------------------------------


class SchedulingStrategyType(str, Enum):
    HEALTH_AWARE = "health_aware"
    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_REMAINING_CAPACITY = "least_remaining_capacity"
    PRIORITY_FAILOVER = "priority_failover"


_STRATEGY_REGISTRY: dict[SchedulingStrategyType, type[BaseSchedulingStrategy]] = {
    SchedulingStrategyType.HEALTH_AWARE: HealthAwareStrategy,
    SchedulingStrategyType.ROUND_ROBIN: RoundRobinStrategy,
    SchedulingStrategyType.LEAST_USED: LeastUsedStrategy,
    SchedulingStrategyType.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinStrategy,
    SchedulingStrategyType.LEAST_REMAINING_CAPACITY: LeastRemainingCapacityStrategy,
    SchedulingStrategyType.PRIORITY_FAILOVER: PriorityFailoverStrategy,
}


def create_strategy(
    strategy_type: SchedulingStrategyType | str,
) -> BaseSchedulingStrategy:
    """Factory: build a fresh strategy instance from an enum value or string."""
    if isinstance(strategy_type, str):
        try:
            strategy_type = SchedulingStrategyType(strategy_type)
        except ValueError as exc:
            available = sorted(item.value for item in SchedulingStrategyType)
            raise UnknownSchedulingStrategyError(
                f"Unknown scheduling strategy {strategy_type!r}. Available strategies: {available}",
                strategy=strategy_type,
                available_strategies=available,
            ) from exc
    cls = _STRATEGY_REGISTRY.get(strategy_type)
    if cls is None:
        available = sorted(item.value for item in SchedulingStrategyType)
        raise UnknownSchedulingStrategyError(
            f"Unknown scheduling strategy {strategy_type!r}. Available strategies: {available}",
            strategy=str(strategy_type),
            available_strategies=available,
        )
    return cls()
