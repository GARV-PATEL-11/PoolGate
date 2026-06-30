from __future__ import annotations

from typing import TYPE_CHECKING

from poolgate.pool.key_pool import APIKeyState
from poolgate.pool.strategies.base import BaseSchedulingStrategy

if TYPE_CHECKING:
    from poolgate.core.config import GroqConfig


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
        return candidates[0]
