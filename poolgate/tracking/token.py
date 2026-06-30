"""TokenTracker — per-model token accounting with rolling windows and daily buckets."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from poolgate.tracking.models import today_str
from poolgate.tracking.rolling_window import RollingWindowCounter

MINUTE = 60
DAY = 24 * 60 * 60


@dataclass
class _ModelTokenState:
    input_window: RollingWindowCounter = field(
        default_factory=lambda: RollingWindowCounter(max_window_seconds=DAY),
    )
    output_window: RollingWindowCounter = field(
        default_factory=lambda: RollingWindowCounter(max_window_seconds=DAY),
    )
    daily: dict[str, dict[str, int]] = field(default_factory=dict)


class TokenTracker:
    """Thread-safe, per-model token accounting."""

    def __init__(self) -> None:
        self._models: dict[str, _ModelTokenState] = {}
        self._lock = threading.Lock()

    def track_input_tokens(self, model: str, tokens: int) -> None:
        self._track(model, tokens, is_input=True)

    def track_output_tokens(self, model: str, tokens: int) -> None:
        self._track(model, tokens, is_input=False)

    def record(self, model: str, tokens_in: int = 0, tokens_out: int = 0) -> None:
        if tokens_in:
            self.track_input_tokens(model, tokens_in)
        if tokens_out:
            self.track_output_tokens(model, tokens_out)

    def get_session_usage(self, session_id: str) -> dict[str, Any]:
        return self.tokens_for_day(session_id)

    def get_key_usage(self, key_id: str) -> dict[str, Any]:
        return self.tokens_for_day(key_id)

    def tokens_in_last_minute(self, model: str) -> int:
        return self._sum_window(model, MINUTE)

    def tokens_in_last_24h(self, model: str) -> int:
        return self._sum_window(model, DAY)

    def remaining_tpm(self, model: str, limit: int) -> int:
        return max(0, limit - self.tokens_in_last_minute(model))

    def remaining_tpd(self, model: str, limit: int) -> int:
        return max(0, limit - self.tokens_in_last_24h(model))

    def tokens_for_day(self, model: str, date: str | None = None) -> dict[str, Any]:
        key = date or today_str()
        with self._lock:
            state = self._models.get(model)
            day = state.daily.get(key, {"in": 0, "out": 0}) if state else {"in": 0, "out": 0}
            return {"date": key, "tokens_in": day["in"], "tokens_out": day["out"]}

    def all_days_for_model(self, model: str) -> list[dict[str, Any]]:
        with self._lock:
            state = self._models.get(model)
            if not state:
                return []
            return [{"date": d, "tokens_in": v["in"], "tokens_out": v["out"]} for d, v in sorted(state.daily.items())]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                model: {
                    "tokens_in": sum(d["in"] for d in state.daily.values()),
                    "tokens_out": sum(d["out"] for d in state.daily.values()),
                }
                for model, state in self._models.items()
            }

    def export_days(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            merged: dict[str, dict[str, Any]] = {}
            for model, state in self._models.items():
                for date, vals in state.daily.items():
                    merged.setdefault(date, {})[model] = {
                        "tokens_in": vals["in"],
                        "tokens_out": vals["out"],
                    }
            return merged

    def load_days(self, days: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            for date, models in days.items():
                for model, vals in models.items():
                    state = self._models.setdefault(model, _ModelTokenState())
                    state.daily[date] = {
                        "in": vals.get("tokens_in", 0),
                        "out": vals.get("tokens_out", 0),
                    }

    def _track(self, model: str, tokens: int, *, is_input: bool) -> None:
        with self._lock:
            state = self._models.setdefault(model, _ModelTokenState())
            window = state.input_window if is_input else state.output_window
            window.add(weight=tokens)
            key = today_str()
            day = state.daily.setdefault(key, {"in": 0, "out": 0})
            day["in" if is_input else "out"] += tokens

    def _sum_window(self, model: str, seconds: int) -> int:
        with self._lock:
            state = self._models.get(model)
        if not state:
            return 0
        return state.input_window.count_since(seconds) + state.output_window.count_since(seconds)
