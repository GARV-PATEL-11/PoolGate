"""DailySnapshotRepository — service-level persistence facade for tracker history."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from poolgate.tracking.persistence import JSONPersistence, Persistence, SQLitePersistence


class PersistableTracker(Protocol):
    def export_days(self) -> dict[str, dict[str, Any]]: ...
    def load_days(self, days: dict[str, dict[str, Any]]) -> None: ...


class DailySnapshotRepository:
    """Coordinates loading and flushing daily tracker state to a persistence backend."""

    def __init__(self, backend: Persistence | None = None) -> None:
        self._backend = backend or JSONPersistence()

    @classmethod
    def json(cls, path: Path | str) -> DailySnapshotRepository:
        return cls(JSONPersistence(path))

    @classmethod
    def sqlite(cls, path: Path | str) -> DailySnapshotRepository:
        return cls(SQLitePersistence(path))

    def load_tracker(self, tracker: PersistableTracker) -> dict[str, dict[str, Any]]:
        days = self._backend.load_all()
        tracker.load_days(days)
        return days

    def flush_tracker(self, tracker: PersistableTracker) -> None:
        self._backend.save_all(tracker.export_days())

    def load_all(self) -> dict[str, dict[str, Any]]:
        return self._backend.load_all()

    def save_day(self, date: str, payload: dict[str, Any]) -> None:
        self._backend.save_day(date, payload)

    def save_all(self, days: dict[str, dict[str, Any]]) -> None:
        self._backend.save_all(days)


# Backwards-compatible alias
PersistenceService = DailySnapshotRepository


class RequestJournal:
    """Appends one JSON line per completed request to a daily JSONL file."""

    def __init__(self, dir_path: Path) -> None:
        self._dir: Path = dir_path
        self._lock = threading.Lock()
        dir_path.mkdir(parents=True, exist_ok=True)

    def _current_path(self) -> Path:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        return self._dir / f"{date_str}.jsonl"

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str) + "\n"
        try:
            with self._lock:
                with open(self._current_path(), "a", encoding="utf-8") as f:
                    f.write(line)
        except OSError:
            pass
