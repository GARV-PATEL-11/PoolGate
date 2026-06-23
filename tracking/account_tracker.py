"""
account_tracker.py
---------------------
Tracks usage per API key — essential since PoolGate rotates across a pool
of keys and needs to know, for each one, both:

  * how close it is to its limit *right now* (rolling 24h window, used to
    pick the least-used key when rotating), and
  * how much it's been used today (calendar day, for a reporting / billing
    style view of each key).

`api_key` here should be a stable identifier (e.g. "key_7", or a hash) —
never store the raw secret in this tracker.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tracking.models import today_str
from tracking.rolling_window import RollingWindowCounter

DAY = 24 * 60 * 60


@dataclass
class _KeyState:
    request_window: RollingWindowCounter = field(
        default_factory=lambda: RollingWindowCounter(max_window_seconds=DAY),
    )
    token_window: RollingWindowCounter = field(
        default_factory=lambda: RollingWindowCounter(max_window_seconds=DAY),
    )
    daily_requests: dict[str, int] = field(default_factory=dict)
    daily_tokens: dict[str, int] = field(default_factory=dict)
    last_used: float | None = None


class AccountTracker:
    """Thread-safe per-API-key usage tracking."""

    def __init__(self) -> None:
        self._keys: dict[str, _KeyState] = {}
        self._lock = threading.Lock()

    def record_use(self, api_key: str, tokens: int = 0) -> None:
        with self._lock:
            state = self._keys.setdefault(api_key, _KeyState())
            state.request_window.add(weight=1)
            if tokens:
                state.token_window.add(weight=tokens)
            state.last_used = time.time()

            key_day = today_str()
            state.daily_requests[key_day] = state.daily_requests.get(key_day, 0) + 1
            if tokens:
                state.daily_tokens[key_day] = state.daily_tokens.get(key_day, 0) + tokens

    # -- rolling 24h view, for rotation decisions ---------------------------

    def requests_last_24h(self, api_key: str) -> int:
        with self._lock:
            state = self._keys.get(api_key)
        return state.request_window.count_since(DAY) if state else 0

    def tokens_last_24h(self, api_key: str) -> int:
        with self._lock:
            state = self._keys.get(api_key)
        return state.token_window.count_since(DAY) if state else 0

    def least_used_key(self, candidate_keys: list[str]) -> str | None:
        """Pick whichever candidate key has used the smallest share of its
        rolling 24h request budget — a simple, fair rotation policy."""
        if not candidate_keys:
            return None
        return min(candidate_keys, key=self.requests_last_24h)

    # -- calendar-day / reporting view ---------------------------------------

    def snapshot(self, api_key: str) -> dict:
        with self._lock:
            state = self._keys.get(api_key)
            if not state:
                return {
                    "api_key": api_key,
                    "requests_today": 0,
                    "tokens_today": 0,
                    "last_used": None,
                }

            today = today_str()
            return {
                "api_key": api_key,
                "requests_today": state.daily_requests.get(today, 0),
                "tokens_today": state.daily_tokens.get(today, 0),
                "last_used": _iso(state.last_used),
            }

    def snapshot_all(self) -> list[dict]:
        with self._lock:
            keys = list(self._keys.keys())
        return [self.snapshot(k) for k in keys]

    # -- persistence hooks ------------------------------------------------------

    def export_days(self) -> dict[str, dict]:
        """Flattened to date -> {api_key: {requests, tokens}} for persistence.py."""
        with self._lock:
            merged: dict[str, dict] = {}
            for api_key, state in self._keys.items():
                dates = set(state.daily_requests) | set(state.daily_tokens)
                for date in dates:
                    merged.setdefault(date, {})[api_key] = {
                        "requests": state.daily_requests.get(date, 0),
                        "tokens": state.daily_tokens.get(date, 0),
                    }
            return merged

    def load_days(self, days: dict[str, dict]) -> None:
        """Restore per-key daily history (output of persistence.load_all())."""
        with self._lock:
            for date, keys in days.items():
                for api_key, vals in keys.items():
                    state = self._keys.setdefault(api_key, _KeyState())
                    state.daily_requests[date] = vals.get("requests", 0)
                    state.daily_tokens[date] = vals.get("tokens", 0)

    def record_request(self, api_key: str, tokens: int = 0) -> None:
        self.record_use(api_key, tokens)

    def get_stats(self, api_key: str | None = None) -> dict | list[dict]:
        return self.snapshot(api_key) if api_key is not None else self.snapshot_all()

    def reset(self) -> None:
        with self._lock:
            self._keys.clear()


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
