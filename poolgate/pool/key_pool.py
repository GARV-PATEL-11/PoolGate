"""API key state tracking and key pool collection."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

from poolgate.core.logger import mask_key
from poolgate.exceptions.keys import APIKeyError
from poolgate.schemas.common.runtime import APIKeyStatus
from poolgate.utils import LatencyTracker, SlidingWindowCounter, now_ts, utc_now


@dataclass
class APIKeyState:
    """Runtime state for one Groq API key. All mutations go through thread-safe methods."""

    key_id: str
    raw_key: str
    masked_key: str
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    created_at: float = field(default_factory=utc_now)
    last_used: float = 0.0

    _rpm_counter: SlidingWindowCounter = field(default_factory=lambda: SlidingWindowCounter(60))
    _rph_counter: SlidingWindowCounter = field(default_factory=lambda: SlidingWindowCounter(3600))
    _rpd_counter: SlidingWindowCounter = field(default_factory=lambda: SlidingWindowCounter(86400))

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    success_count: int = 0
    failure_count: int = 0
    last_failure: float = 0.0
    last_success: float = 0.0
    consecutive_429s: int = 0
    consecutive_failures: int = 0
    cooldown_until: float = 0.0

    _latency_tracker: LatencyTracker = field(default_factory=LatencyTracker)

    active_requests: int = 0
    max_parallel_requests: int = 10

    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_key(cls, key_id: str, raw_key: str, max_parallel: int = 10) -> APIKeyState:
        return cls(
            key_id=key_id,
            raw_key=raw_key,
            masked_key=mask_key(raw_key),
            max_parallel_requests=max_parallel,
        )

    @property
    def requests_per_minute(self) -> int:
        return self._rpm_counter.count()

    @property
    def requests_per_hour(self) -> int:
        return self._rph_counter.count()

    @property
    def requests_per_day(self) -> int:
        return self._rpd_counter.count()

    @property
    def failure_rate(self) -> float:
        total = self.success_count + self.failure_count
        return (self.failure_count / total) if total > 0 else 0.0

    @property
    def average_latency(self) -> float:
        return self._latency_tracker.average()

    @property
    def p95_latency(self) -> float:
        return self._latency_tracker.p95()

    @property
    def is_cooling_down(self) -> bool:
        return now_ts() < self.cooldown_until

    @property
    def is_available(self) -> bool:
        return (
            self.status == APIKeyStatus.ACTIVE
            and not self.is_cooling_down
            and self.active_requests < self.max_parallel_requests
        )

    def record_request_start(self) -> None:
        with self._lock:
            self.active_requests += 1
            self.last_used = utc_now()
            self._rpm_counter.record()
            self._rph_counter.record()
            self._rpd_counter.record()

    def record_request_end(self, latency: float, tokens_in: int, tokens_out: int) -> None:
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.success_count += 1
            self.last_success = utc_now()
            self.consecutive_429s = 0
            self.consecutive_failures = 0
            self.input_tokens += tokens_in
            self.output_tokens += tokens_out
            self.total_tokens += tokens_in + tokens_out
            self._latency_tracker.record(latency)
            if self.status == APIKeyStatus.RATE_LIMITED:
                self.status = APIKeyStatus.ACTIVE

    def record_failure(self, is_rate_limit: bool = False, cooldown_secs: float = 60.0) -> None:
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.failure_count += 1
            self.consecutive_failures += 1
            self.last_failure = utc_now()
            if is_rate_limit:
                self.consecutive_429s += 1
                self.status = APIKeyStatus.RATE_LIMITED
                self.cooldown_until = now_ts() + cooldown_secs
            else:
                self.consecutive_429s = 0

    def mark_disabled(self) -> None:
        with self._lock:
            self.status = APIKeyStatus.DISABLED

    def mark_failed(self) -> None:
        with self._lock:
            self.status = APIKeyStatus.FAILED

    def reset_cooldown(self) -> None:
        with self._lock:
            self.cooldown_until = 0.0
            if self.status in (APIKeyStatus.RATE_LIMITED, APIKeyStatus.COOLDOWN):
                self.status = APIKeyStatus.ACTIVE

    def health_score(self, max_rpm: int = 30, latency_threshold: float = 3.0) -> float:
        if not self.is_available:
            return float("-inf")
        rpm_penalty = max(0.0, (self.requests_per_minute / max(max_rpm, 1)) * 4.0)
        active_penalty = (self.active_requests / max(self.max_parallel_requests, 1)) * 3.0
        latency_penalty = max(0.0, (self.average_latency - latency_threshold)) * 0.5
        failure_penalty = self.failure_rate * 3.0
        recent_failure_penalty = 0.0
        if self.last_failure > 0:
            secs_ago = utc_now() - self.last_failure
            if secs_ago < 300:
                recent_failure_penalty = max(0.0, (300 - secs_ago) / 300) * 2.0
        cooldown_penalty = 1.0 if self.is_cooling_down else 0.0
        return (
            10.0
            - rpm_penalty
            - active_penalty
            - latency_penalty
            - failure_penalty
            - recent_failure_penalty
            - cooldown_penalty
        )

    def __repr__(self) -> str:
        return (
            f"APIKeyState(id={self.key_id!r}, masked={self.masked_key!r}, "
            f"status={self.status.value}, rpm={self.requests_per_minute}, "
            f"active={self.active_requests}, score={self.health_score():.2f})"
        )


APIKey = APIKeyState


class KeyPool:
    """Thread-safe collection of managed API key states."""

    def __init__(
        self,
        keys: Iterable[str | APIKeyState] | None = None,
        *,
        max_parallel: int = 10,
    ) -> None:
        self._keys: dict[str, APIKeyState] = {}
        self._counter = 0
        self._max_parallel = max_parallel
        self._lock = threading.RLock()
        for key in keys or ():
            if isinstance(key, APIKeyState):
                self.add_key_state(key)
            else:
                self.add_key(key)

    def add_key(
        self,
        raw_key: str,
        *,
        key_id: str | None = None,
        max_parallel: int | None = None,
    ) -> APIKeyState:
        with self._lock:
            resolved_id = key_id or self._next_key_id()
            state = APIKeyState.from_key(
                key_id=resolved_id,
                raw_key=raw_key,
                max_parallel=max_parallel or self._max_parallel,
            )
            self._keys[resolved_id] = state
            return state

    def add_key_state(self, key: APIKeyState) -> APIKeyState:
        with self._lock:
            self._keys[key.key_id] = key
            return key

    def remove_key(self, key_id: str) -> APIKeyState | None:
        with self._lock:
            return self._keys.pop(key_id, None)

    def enable_key(self, key_id: str) -> None:
        with self._lock:
            key = self._require_key(key_id)
            key.status = APIKeyStatus.ACTIVE
            key.cooldown_until = 0.0

    def disable_key(self, key_id: str) -> None:
        self._require_key(key_id).mark_disabled()

    def mark_unhealthy(
        self,
        key_id: str,
        *,
        failed: bool = False,
        cooldown_secs: float = 60.0,
    ) -> None:
        key = self._require_key(key_id)
        if failed:
            key.mark_failed()
        else:
            key.record_failure(is_rate_limit=True, cooldown_secs=cooldown_secs)

    def get_eligible_keys(self, *, capability: str | None = None) -> list[APIKeyState]:
        with self._lock:
            return [key for key in self._keys.values() if key.is_available]

    def get_key(self, key_id: str | None = None) -> APIKeyState | None:
        with self._lock:
            if key_id is not None:
                return self._keys.get(key_id)
            eligible = self.get_eligible_keys()
            return max(eligible, key=lambda key: key.health_score()) if eligible else None

    def list_keys(self) -> list[APIKeyState]:
        with self._lock:
            return list(self._keys.values())

    def _next_key_id(self) -> str:
        while True:
            key_id = f"key_{self._counter}"
            self._counter += 1
            if key_id not in self._keys:
                return key_id

    def _require_key(self, key_id: str) -> APIKeyState:
        with self._lock:
            key = self._keys.get(key_id)
        if key is None:
            raise APIKeyError(f"Unknown API key id: {key_id!r}")
        return key


GroqKeyPool = KeyPool
