"""UsageTracker — global request/token counters with per-calendar-day buckets."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from poolgate.tracking.models import DailyBucket, today_str


@dataclass
class GlobalUsage:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests


class UsageTracker:
    """Thread-safe global counter plus calendar-day breakdown."""

    def __init__(self) -> None:
        self._lifetime = GlobalUsage()
        self._days: dict[str, DailyBucket] = {}
        self._lock = threading.Lock()

    def record_success(self, tokens_in: int, tokens_out: int, retried: bool = False) -> None:
        with self._lock:
            self._lifetime.total_requests += 1
            self._lifetime.successful_requests += 1
            self._lifetime.input_tokens += tokens_in
            self._lifetime.output_tokens += tokens_out
            if retried:
                self._lifetime.total_retries += 1
            day = self._bucket_for_today()
            day.requests += 1
            day.successful_requests += 1
            day.tokens_in += tokens_in
            day.tokens_out += tokens_out

    def on_request_start(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def on_request_end(self, tokens_in: int = 0, tokens_out: int = 0, retried: bool = False, **_kwargs: Any) -> None:
        self.record_success(tokens_in, tokens_out, retried=retried)

    def on_request_failure(self, retried: bool = False, **_kwargs: Any) -> None:
        self.record_failure(retried=retried)

    def get_key_usage(self, _key_id: str | None = None) -> dict[str, Any]:
        return self.snapshot()

    def record_failure(self, retried: bool = False) -> None:
        with self._lock:
            self._lifetime.total_requests += 1
            self._lifetime.failed_requests += 1
            if retried:
                self._lifetime.total_retries += 1
            day = self._bucket_for_today()
            day.requests += 1
            day.failed_requests += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            u = self._lifetime
            return {
                "total_requests": u.total_requests,
                "successful_requests": u.successful_requests,
                "failed_requests": u.failed_requests,
                "total_retries": u.total_retries,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "total_tokens": u.total_tokens,
                "success_rate": round(u.success_rate, 4),
            }

    def snapshot_for_day(self, date: str | None = None) -> dict[str, Any]:
        key = date or today_str()
        with self._lock:
            bucket = self._days.get(key, DailyBucket(date=key))
            return bucket.to_dict()

    def snapshot_all_days(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._days[k].to_dict() for k in sorted(self._days.keys())]

    def export_days(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: v.to_dict() for k, v in self._days.items()}

    def load_days(self, days: dict[str, Any]) -> None:
        for key, payload in days.items():
            if "date" not in payload:
                payload = {**payload, "date": key}
            self._days[key] = DailyBucket.from_dict(payload)

    def _bucket_for_today(self) -> DailyBucket:
        key = today_str()
        if key not in self._days:
            self._days[key] = DailyBucket(date=key)
        return self._days[key]
