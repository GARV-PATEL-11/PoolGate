"""Session lifecycle and per-session usage accounting."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from poolgate.exceptions.output import SessionExpiredError
from poolgate.utils import LatencyTracker, utc_now


@dataclass
class ModelUsageStat:
    request_count: int = 0
    tokens: int = 0
    total_latency: float = 0.0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.request_count if self.request_count else 0.0


@dataclass
class SessionUsageTracker:
    session_id: str
    created_at: float = field(default_factory=utc_now)
    expires_at: float = 0.0
    last_activity: float = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _latency_tracker: LatencyTracker = field(default_factory=LatencyTracker)
    _model_stats: dict[str, ModelUsageStat] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def average_latency(self) -> float:
        return self._latency_tracker.average()

    @property
    def p95_latency(self) -> float:
        return self._latency_tracker.p95()

    @property
    def is_expired(self) -> bool:
        return utc_now() > self.expires_at

    @property
    def model_usage(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                model: {
                    "request_count": stat.request_count,
                    "tokens": stat.tokens,
                    "avg_latency": round(stat.avg_latency, 4),
                }
                for model, stat in self._model_stats.items()
            }

    def record_success(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency: float,
        retried: bool = False,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.successful_requests += 1
            self.input_tokens += tokens_in
            self.output_tokens += tokens_out
            self.last_activity = utc_now()
            self._latency_tracker.record(latency)
            if retried:
                self.retries += 1
            stat = self._model_stats.setdefault(model, ModelUsageStat())
            stat.request_count += 1
            stat.tokens += tokens_in + tokens_out
            stat.total_latency += latency

    def record_failure(self, retried: bool = False) -> None:
        with self._lock:
            self.total_requests += 1
            self.failed_requests += 1
            self.last_activity = utc_now()
            if retried:
                self.retries += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_activity": self.last_activity,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "retries": self.retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "average_latency": round(self.average_latency, 4),
            "p95_latency": round(self.p95_latency, 4),
            "model_usage": self.model_usage,
            "metadata": self.metadata,
        }


class SessionManager:
    """Thread-safe session registry with TTL expiry."""

    def __init__(self, session_ttl_hours: int = 24) -> None:
        self._ttl_seconds = session_ttl_hours * 3600
        self._sessions: dict[str, SessionUsageTracker] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionUsageTracker:
        with self._lock:
            if session_id:
                existing = self._sessions.get(session_id)
                if existing:
                    if existing.is_expired:
                        del self._sessions[session_id]
                        raise SessionExpiredError(session_id)
                    return existing
            sid = session_id or str(uuid.uuid4())
            tracker = SessionUsageTracker(
                session_id=sid,
                expires_at=utc_now() + self._ttl_seconds,
                metadata=metadata or {},
            )
            self._sessions[sid] = tracker
            return tracker

    def get(self, session_id: str) -> SessionUsageTracker | None:
        with self._lock:
            tracker = self._sessions.get(session_id)
            if tracker and tracker.is_expired:
                del self._sessions[session_id]
                return None
            return tracker

    def expire_old_sessions(self) -> int:
        with self._lock:
            expired = [sid for sid, tracker in self._sessions.items() if tracker.is_expired]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for tracker in self._sessions.values() if not tracker.is_expired)

    def get_stats(self, session_id: str) -> dict[str, Any] | None:
        tracker = self.get(session_id)
        return tracker.to_dict() if tracker else None
