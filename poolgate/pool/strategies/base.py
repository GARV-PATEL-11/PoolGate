"""Base scheduling strategy interface and strategy type enum."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from poolgate.exceptions.request import UnknownSchedulingStrategyError
from poolgate.pool.key_pool import APIKeyState

if TYPE_CHECKING:
    from poolgate.core.config import GroqConfig


class BaseSchedulingStrategy(ABC):
    """Common interface every scheduling strategy must implement."""

    @abstractmethod
    def select(
        self,
        candidates: list[APIKeyState],
        all_keys: list[APIKeyState],
        config: GroqConfig,
    ) -> APIKeyState:
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__


SchedulingStrategy = BaseSchedulingStrategy


class SchedulingStrategyType(str, Enum):
    HEALTH_AWARE = "health_aware"
    ROUND_ROBIN = "round_robin"
    LEAST_USED = "least_used"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_REMAINING_CAPACITY = "least_remaining_capacity"
    PRIORITY_FAILOVER = "priority_failover"


def create_strategy(strategy_type: SchedulingStrategyType | str) -> BaseSchedulingStrategy:
    """Factory: build a fresh strategy instance from an enum value or string."""
    from poolgate.pool.strategies.health_aware import HealthAwareStrategy
    from poolgate.pool.strategies.least_used import LeastRemainingCapacityStrategy, LeastUsedStrategy
    from poolgate.pool.strategies.priority_failover import PriorityFailoverStrategy
    from poolgate.pool.strategies.round_robin import RoundRobinStrategy
    from poolgate.pool.strategies.weighted import WeightedRoundRobinStrategy

    _REGISTRY: dict[SchedulingStrategyType, type[BaseSchedulingStrategy]] = {
        SchedulingStrategyType.HEALTH_AWARE: HealthAwareStrategy,
        SchedulingStrategyType.ROUND_ROBIN: RoundRobinStrategy,
        SchedulingStrategyType.LEAST_USED: LeastUsedStrategy,
        SchedulingStrategyType.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinStrategy,
        SchedulingStrategyType.LEAST_REMAINING_CAPACITY: LeastRemainingCapacityStrategy,
        SchedulingStrategyType.PRIORITY_FAILOVER: PriorityFailoverStrategy,
    }

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

    cls = _REGISTRY.get(strategy_type)
    if cls is None:
        available = sorted(item.value for item in SchedulingStrategyType)
        raise UnknownSchedulingStrategyError(
            f"Unknown scheduling strategy {strategy_type!r}.",
            strategy=str(strategy_type),
            available_strategies=available,
        )
    return cls()
