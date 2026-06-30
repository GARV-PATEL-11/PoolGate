"""RequestTracker — rolling RPM/RPD request counters per scope."""

from __future__ import annotations

import threading

from typing import Any

from poolgate.tracking.rolling_window import RollingWindowCounter

MINUTE = 60
DAY = 24 * 60 * 60


class RequestTracker:
    """Thread-safe rolling RPM/RPD counter, one per scope."""

    def __init__(self) -> None:
        self._scopes: dict[str, RollingWindowCounter] = {}
        self._lock = threading.Lock()

    def record_request(self, scope: str = "global") -> None:
        with self._lock:
            counter = self._scopes.setdefault(scope, RollingWindowCounter(max_window_seconds=DAY))
        counter.add(weight=1)

    def record(self, scope: str = "global") -> None:
        self.record_request(scope)

    def update(self, scope: str = "global") -> None:
        self.record_request(scope)

    def get_request(self, scope: str = "global") -> dict[str, Any]:
        return self.snapshot(scope)

    def requests_per_minute(self, scope: str = "global") -> int:
        return self._count(scope, MINUTE)

    def requests_per_day(self, scope: str = "global") -> int:
        return self._count(scope, DAY)

    def remaining_rpm(self, scope: str, limit: int) -> int:
        return max(0, limit - self.requests_per_minute(scope))

    def remaining_rpd(self, scope: str, limit: int) -> int:
        return max(0, limit - self.requests_per_day(scope))

    def seconds_until_rpm_frees_up(self, scope: str = "global") -> float:
        counter = self._scopes.get(scope)
        return counter.reset_in_seconds(MINUTE) if counter else 0.0

    def seconds_until_rpd_frees_up(self, scope: str = "global") -> float:
        counter = self._scopes.get(scope)
        return counter.reset_in_seconds(DAY) if counter else 0.0

    def snapshot(self, scope: str = "global") -> dict[str, Any]:
        return {
            "scope": scope,
            "requests_per_minute": self.requests_per_minute(scope),
            "requests_per_day": self.requests_per_day(scope),
        }

    def _count(self, scope: str, seconds: int) -> int:
        with self._lock:
            counter = self._scopes.get(scope)
        return counter.count_since(seconds) if counter else 0
