"""Durable storage for tracking data — JSON and SQLite backends."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

from poolgate.exceptions.persistence import PersistenceError


class Persistence(ABC):
    @abstractmethod
    def load_all(self) -> dict[str, dict[str, Any]]: ...

    @abstractmethod
    def save_day(self, date: str, payload: dict[str, Any]) -> None: ...

    @abstractmethod
    def save_all(self, days: dict[str, dict[str, Any]]) -> None: ...


class JSONPersistence(Persistence):
    """Atomic JSON-file backend. Writes are temp-file-then-rename to prevent corruption."""

    def __init__(self, path: Path | str = "usage.json") -> None:
        self._path: Path = Path(path)
        self._lock = threading.Lock()
        if not self._path.exists():
            self._write({})

    def load_all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return self._read_unlocked()

    def save_day(self, date: str, payload: dict[str, Any]) -> None:
        with self._lock:
            data = self._read_unlocked()
            data[date] = payload
            self._write(data)

    def save_all(self, days: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            self._write(days)

    def _read_unlocked(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path, encoding="utf-8") as f:
                return cast(dict[str, dict[str, Any]], json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            raise PersistenceError(
                f"Failed to load JSON persistence file {str(self._path)!r}: {exc}",
                backend="json",
                path=str(self._path),
            ) from exc

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        directory = self._path.resolve().parent
        try:
            fd, tmp_path_str = tempfile.mkstemp(dir=str(directory), prefix=".usage_", suffix=".tmp")
        except OSError as exc:
            raise PersistenceError(
                f"Failed to create temp persistence file beside {str(self._path)!r}: {exc}",
                backend="json",
                path=str(self._path),
            ) from exc
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            tmp_path.replace(self._path)
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            raise PersistenceError(
                f"Failed to write JSON persistence file {str(self._path)!r}: {exc}",
                backend="json",
                path=str(self._path),
            ) from exc


class SQLitePersistence(Persistence):
    """Single-table SQLite backend."""

    def __init__(self, path: Path | str = "usage.db") -> None:
        self._path: Path = Path(path)
        self._lock = threading.Lock()
        try:
            with sqlite3.connect(str(self._path)) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS daily_usage (date TEXT PRIMARY KEY, payload TEXT NOT NULL)",
                )
        except sqlite3.Error as exc:
            raise PersistenceError(
                f"Failed to initialize SQLite persistence database {str(self._path)!r}: {exc}",
                backend="sqlite",
                path=str(self._path),
            ) from exc

    def load_all(self) -> dict[str, dict[str, Any]]:
        try:
            with self._lock, sqlite3.connect(str(self._path)) as conn:
                rows = conn.execute("SELECT date, payload FROM daily_usage").fetchall()
                return {date: cast(dict[str, Any], json.loads(payload)) for date, payload in rows}
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise PersistenceError(
                f"Failed to load SQLite persistence database {str(self._path)!r}: {exc}",
                backend="sqlite",
                path=str(self._path),
            ) from exc

    def save_day(self, date: str, payload: dict[str, Any]) -> None:
        try:
            with self._lock, sqlite3.connect(str(self._path)) as conn:
                conn.execute(
                    "INSERT INTO daily_usage (date, payload) VALUES (?, ?) "
                    "ON CONFLICT(date) DO UPDATE SET payload = excluded.payload",
                    (date, json.dumps(payload)),
                )
        except (sqlite3.Error, TypeError) as exc:
            raise PersistenceError(
                f"Failed to save SQLite persistence day {date!r}: {exc}",
                backend="sqlite",
                path=str(self._path),
            ) from exc

    def save_all(self, days: dict[str, dict[str, Any]]) -> None:
        try:
            with self._lock, sqlite3.connect(str(self._path)) as conn:
                conn.executemany(
                    "INSERT INTO daily_usage (date, payload) VALUES (?, ?) "
                    "ON CONFLICT(date) DO UPDATE SET payload = excluded.payload",
                    [(d, json.dumps(p)) for d, p in days.items()],
                )
        except (sqlite3.Error, TypeError) as exc:
            raise PersistenceError(
                f"Failed to save SQLite persistence database {str(self._path)!r}: {exc}",
                backend="sqlite",
                path=str(self._path),
            ) from exc
