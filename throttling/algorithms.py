"""Core throttle algorithms: sliding window counter, token bucket, concurrency counter."""

from __future__ import annotations

import threading
import time
from collections import deque


class SlidingWindowCounter:
    """Thread-safe 60-second sliding window request counter."""

    def __init__(self, window_secs: float = 60.0) -> None:
        self._window = window_secs
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def try_acquire(self, limit: int) -> bool:
        """Atomically check limit and record timestamp. Returns True if allowed."""
        now = time.monotonic()
        with self._lock:
            cutoff = now - self._window
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= limit:
                return False
            self._timestamps.append(now)
            return True

    def count(self) -> int:
        """Current count within the window (read-only, for monitoring)."""
        now = time.monotonic()
        with self._lock:
            cutoff = now - self._window
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            return len(self._timestamps)


class TokenBucket:
    """Thread-safe token bucket with lazy refill for smooth per-model rate limiting."""

    def __init__(self, max_rpm: int) -> None:
        self._rate_per_sec = max_rpm / 60.0
        self._capacity = float(max_rpm)
        self._tokens = float(max_rpm)  # start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        """Refill lazily, then consume one token. Returns True if allowed."""
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_sec)
            self._last_refill = now
            if self._tokens < 1.0:
                return False
            self._tokens -= 1.0
            return True

    def available(self) -> float:
        """Current token estimate (for monitoring; not atomically consistent with try_acquire)."""
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last_refill
            return min(self._capacity, self._tokens + elapsed * self._rate_per_sec)


class ConcurrencyCounter:
    """Thread-safe counter for tracking in-flight requests."""

    def __init__(self) -> None:
        self._count = 0
        self._lock = threading.Lock()

    def try_acquire(self, limit: int) -> bool:
        with self._lock:
            if self._count >= limit:
                return False
            self._count += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._count > 0:
                self._count -= 1

    def count(self) -> int:
        with self._lock:
            return self._count
