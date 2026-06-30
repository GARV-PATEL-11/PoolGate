"""ISchedulingStrategy — DIP contract for key-selection strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from poolgate.pool.key_pool import APIKeyState


@runtime_checkable
class ISchedulingStrategy(Protocol):
    def select(self, candidates: list["APIKeyState"]) -> "APIKeyState": ...
