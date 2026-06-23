"""Service-level persistence facade for tracker history."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from tracking.persistence import JSONPersistence, Persistence, SQLitePersistence


class PersistableTracker(Protocol):

    def export_days(self) -> dict[str, dict]: ...

    def load_days(self, days: dict[str, dict]) -> None: ...


class PersistenceService:
    """Coordinates loading and flushing daily tracker state."""

    def __init__(self, backend: Persistence | None = None) -> None:
        self._backend = backend or JSONPersistence()

    @classmethod
    def json(cls, path: Path | str) -> PersistenceService:
        return cls(JSONPersistence(path))

    @classmethod
    def sqlite(cls, path: Path | str) -> PersistenceService:
        return cls(SQLitePersistence(path))

    def load_tracker(self, tracker: PersistableTracker) -> dict[str, dict]:
        days = self._backend.load_all()
        tracker.load_days(days)
        return days

    def flush_tracker(self, tracker: PersistableTracker) -> None:
        self._backend.save_all(tracker.export_days())

    def load_all(self) -> dict[str, dict]:
        return self._backend.load_all()

    def save_day(self, date: str, payload: dict) -> None:
        self._backend.save_day(date, payload)

    def save_all(self, days: dict[str, dict]) -> None:
        self._backend.save_all(days)


class RequestJournal:
    """
    Appends one JSON line per completed request to a daily JSONL file.

    Files are written to ``<dir_path>/YYYY-MM-DD.jsonl`` and rotate
    automatically at midnight UTC.  Each line is a self-contained JSON object
    with all execution details for one request.

    Thread-safe.  Writes are unbuffered so entries survive crashes.
    """

    def __init__(self, dir_path: Path) -> None:
        self._dir: Path = dir_path
        self._lock = threading.Lock()
        dir_path.mkdir(parents=True, exist_ok=True)

    def _current_path(self) -> Path:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        return self._dir / f"{date_str}.jsonl"

    def append(self, record: dict[str, Any]) -> None:
        """Write one JSON line.  Silently drops on I/O error to never block callers."""
        line = json.dumps(record, default=str) + "\n"
        try:
            with self._lock:
                with open(self._current_path(), "a", encoding="utf-8") as f:
                    f.write(line)
        except OSError:
            pass
