"""Rolling window counter for sliding rate-limit windows."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

DEFAULT_MAX_WINDOW_SECONDS = 24 * 60 * 60


@dataclass
class _Entry:
    timestamp: float
    weight: int = 1


class RollingWindowCounter:
    """Thread-safe sliding-window counter."""

    def __init__(self, max_window_seconds: int = DEFAULT_MAX_WINDOW_SECONDS) -> None:
        self._entries: deque[_Entry] = deque()
        self._lock = threading.Lock()
        self._max_window_seconds = max_window_seconds

    def add(self, weight: int = 1, *, now: float | None = None) -> None:
        ts = now if now is not None else time.time()
        with self._lock:
            self._entries.append(_Entry(timestamp=ts, weight=weight))
            self._evict(ts)

    def count_since(self, seconds: int, *, now: float | None = None) -> int:
        ts = now if now is not None else time.time()
        cutoff = ts - seconds
        with self._lock:
            self._evict(ts)
            return sum(e.weight for e in self._entries if e.timestamp >= cutoff)

    def remaining(self, limit: int, seconds: int, *, now: float | None = None) -> int:
        return max(0, limit - self.count_since(seconds, now=now))

    def reset_in_seconds(self, seconds: int, *, now: float | None = None) -> float:
        ts = now if now is not None else time.time()
        cutoff = ts - seconds
        with self._lock:
            in_window = [e for e in self._entries if e.timestamp >= cutoff]
            if not in_window:
                return 0.0
            oldest = min(e.timestamp for e in in_window)
            return max(0.0, (oldest + seconds) - ts)

    def _evict(self, now: float) -> None:
        cutoff = now - self._max_window_seconds
        while self._entries and self._entries[0].timestamp < cutoff:
            self._entries.popleft()


class RollingWindow(RollingWindowCounter):
    """Alias exposing add/get_sum/get_count/prune."""

    def get_sum(self, seconds: int | None = None, *, now: float | None = None) -> int:
        return self.count_since(seconds or self._max_window_seconds, now=now)

    def get_count(self, seconds: int | None = None, *, now: float | None = None) -> int:
        return self.get_sum(seconds, now=now)

    def prune(self, *, now: float | None = None) -> None:
        ts = now if now is not None else time.time()
        with self._lock:
            self._evict(ts)
