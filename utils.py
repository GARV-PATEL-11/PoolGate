"""
Shared utility helpers.
Kept minimal — only what is used across multiple modules.
"""

from __future__ import annotations

import math
import time
from collections import deque
from threading import Lock


class SlidingWindowCounter:
    """
    Thread-safe sliding-window request counter.
    Counts events in the last `window_seconds`.
    """

    def __init__(self, window_seconds: float) -> None:
        self._window = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def record(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._timestamps.append(now)
            self._evict(now)

    def count(self) -> int:
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            return len(self._timestamps)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()


class LatencyTracker:
    """
    Thread-safe rolling latency tracker.
    Keeps last `max_samples` measurements and exposes average + p95.
    """

    def __init__(self, max_samples: int = 200) -> None:
        self._samples: deque[float] = deque(maxlen=max_samples)
        self._lock = Lock()

    def record(self, latency: float) -> None:
        with self._lock:
            self._samples.append(latency)

    def average(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            return sum(self._samples) / len(self._samples)

    def p95(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            sorted_samples = sorted(self._samples)
            idx = math.ceil(0.95 * len(sorted_samples)) - 1
            return sorted_samples[max(idx, 0)]


def now_ts() -> float:
    """Return current monotonic timestamp (seconds)."""
    return time.monotonic()


def utc_now() -> float:
    """Return current wall-clock UTC time as float (seconds since epoch)."""
    return time.time()


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
