"""ITracker — DIP contract for usage/analytics trackers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ITracker(Protocol):
    def record_success(self, **kwargs: Any) -> None: ...
    def record_failure(self, **kwargs: Any) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...
