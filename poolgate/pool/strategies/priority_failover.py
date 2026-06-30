from __future__ import annotations

from typing import TYPE_CHECKING, Any

from poolgate.pool.key_pool import APIKeyState
from poolgate.pool.strategies.base import BaseSchedulingStrategy

if TYPE_CHECKING:
    from poolgate.core.config import GroqConfig


class PriorityFailoverStrategy(BaseSchedulingStrategy):
    """Always prefers the primary key; falls back to backups when unavailable."""

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
