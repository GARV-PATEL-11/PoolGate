"""ThrottleManager — orchestrates pre-flight RPM checks and post-flight quota updates."""

from __future__ import annotations

from typing import Any

from poolgate.throttling.quota import QuotaTracker
from poolgate.throttling.sliding_window import SlidingWindowBucket


class ThrottleManager:
    """
    Centralizes throttle enforcement. Pre-flight: checks RPM budget.
    Post-flight: updates provider quota from response headers.
    """

    def __init__(self, rpm_limit: int = 30, window_seconds: int = 60) -> None:
        self._rpm_limit = rpm_limit
        self._window = SlidingWindowBucket(window_seconds)
        self._quota = QuotaTracker()

    def check_rpm(self) -> bool:
        """Return True if a new request is within the RPM budget."""
        return self._window.count() < self._rpm_limit

    def record_request(self) -> None:
        self._window.record()

    def update_from_headers(self, model: str, headers: dict[str, Any]) -> None:
        self._quota.update_from_headers(model, headers)

    def is_quota_exhausted(self, model: str) -> bool:
        return self._quota.is_exhausted(model)

    def quota_snapshot(self) -> list[dict[str, Any]]:
        return self._quota.snapshot_all()
