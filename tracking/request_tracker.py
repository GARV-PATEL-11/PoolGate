"""
request_tracker.py
---------------------
Tracks request *counts* (not tokens) using rolling windows — requests_per_minute and requests_per_day
limits are enforced by providers as sliding windows measured from now, not
as counters that reset at :00 or at midnight.

Supports multiple scopes (e.g. one bucket per model, or per API key)
because Groq enforces requests_per_minute/requests_per_day per model, and a caller may also want a
per-key view for rotation decisions. This tracker doesn't care what a
"scope" string means — every caller decides (a model name, a key id,
"global", etc).

Not persisted to disk on purpose: a rolling window is only meaningful
relative to "now", so there's nothing useful to write to a daily history
file here. If you want a calendar-day request count for reporting, that
already lives in usage_tracker.py.
"""

from __future__ import annotations

import threading

from tracking.rolling_window import RollingWindowCounter

MINUTE = 60
DAY = 24 * 60 * 60


class RequestTracker:
    """Thread-safe rolling requests_per_minute / requests_per_day counter, one per scope."""

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

    def get_request(self, scope: str = "global") -> dict:
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

    def snapshot(self, scope: str = "global") -> dict:
        return {
            "scope": scope,
            "requests_per_minute": self.requests_per_minute(scope),
            "requests_per_day": self.requests_per_day(scope),
        }

    def _count(self, scope: str, seconds: int) -> int:
        with self._lock:
            counter = self._scopes.get(scope)
        return counter.count_since(seconds) if counter else 0
