"""
persistence.py
------------------
Durable storage for tracking data. The contract is simple and
storage-engine agnostic: any tracker that keeps calendar-day history
(usage_tracker, token_tracker, account_tracker) exports a
`dict[date_str, payload]` — one calendar day per key, oldest day first —
and persistence.py is responsible for getting that dict to and from disk
so the full history survives a restart, from the very first day onward.

Ships with a JSON-file backend by default (good enough for a single
PoolGate instance). Swap in SQLitePersistence, or a future Redis-backed
class, by implementing the same three methods — nothing else in
tracking/ needs to change.

Each tracker that wants persistence should get its own instance pointed
at its own file/table (e.g. "usage.json", "token_usage.json",
"account_usage.json") — this module doesn't care what the payloads
contain, only that they're keyed by date.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
from abc import ABC, abstractmethod

from exceptions.persistence import PersistenceError


class Persistence(ABC):
	"""Storage contract every backend must satisfy."""

	@abstractmethod
	def load_all(self) -> dict[str, dict]:
		"""Returns every stored day, e.g. {'2026-06-01': {...}, ..., '2026-06-18': {...}}."""

	@abstractmethod
	def save_day(self, date: str, payload: dict) -> None:
		"""Persist (or overwrite) a single day's data."""

	@abstractmethod
	def save_all(self, days: dict[str, dict]) -> None:
		"""Bulk overwrite — used for periodic flushes or on shutdown."""


class JSONPersistence(Persistence):
	"""
	Stores everything in one JSON file, keyed by calendar day:

		{
		  "2026-05-01": {"requests": 120, "tokens_in": 50000, ...},
		  "2026-05-02": {...},
		  ...
		  "2026-06-18": {...}
		}

	Writes are atomic (write to a temp file, then os.replace) so a crash
	mid-write can't corrupt months of history.
	"""

	def __init__(self, path: str = "usage.json") -> None:
		self._path = path
		self._lock = threading.Lock()
		if not os.path.exists(self._path):
			self._write({})

	def load_all(self) -> dict[str, dict]:
		with self._lock:
			return self._read_unlocked()

	def save_day(self, date: str, payload: dict) -> None:
		with self._lock:
			data = self._read_unlocked()
			data[date] = payload
			self._write(data)

	def save_all(self, days: dict[str, dict]) -> None:
		with self._lock:
			self._write(days)

	# -- internals ----------------------------------------------------------

	def _read_unlocked(self) -> dict[str, dict]:
		if not os.path.exists(self._path):
			return {}
		try:
			with open(self._path, encoding="utf-8") as f:
				return json.load(f)
		except (OSError, json.JSONDecodeError) as exc:
			raise PersistenceError(
				f"Failed to load JSON persistence file {self._path!r}: {exc}",
				backend="json",
				path=self._path,
				) from exc

	def _write(self, data: dict[str, dict]) -> None:
		directory = os.path.dirname(os.path.abspath(self._path)) or "."
		try:
			fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".usage_", suffix=".tmp")
		except OSError as exc:
			raise PersistenceError(
				f"Failed to create temp persistence file beside {self._path!r}: {exc}",
				backend="json",
				path=self._path,
				) from exc
		try:
			with os.fdopen(fd, "w", encoding="utf-8") as f:
				json.dump(data, f, indent=2, sort_keys=True)
			os.replace(tmp_path, self._path)
		except OSError as exc:
			if os.path.exists(tmp_path):
				os.remove(tmp_path)
			raise PersistenceError(
				f"Failed to write JSON persistence file {self._path!r}: {exc}",
				backend="json",
				path=self._path,
				) from exc


class SQLitePersistence(Persistence):
	"""
	Same contract, backed by a single-table SQLite db instead of one big
	JSON blob — worth switching to once daily history grows large enough
	that rewriting the whole JSON file on every save gets wasteful.
	"""

	def __init__(self, path: str = "usage.db") -> None:
		self._path = path
		self._lock = threading.Lock()
		try:
			with sqlite3.connect(self._path) as conn:
				conn.execute(
					"CREATE TABLE IF NOT EXISTS daily_usage ("
					"date TEXT PRIMARY KEY, payload TEXT NOT NULL)",
					)
		except sqlite3.Error as exc:
			raise PersistenceError(
				f"Failed to initialize SQLite persistence database {self._path!r}: {exc}",
				backend="sqlite",
				path=self._path,
				) from exc

	def load_all(self) -> dict[str, dict]:
		try:
			with self._lock, sqlite3.connect(self._path) as conn:
				rows = conn.execute("SELECT date, payload FROM daily_usage").fetchall()
				return {date: json.loads(payload) for date, payload in rows}
		except (sqlite3.Error, json.JSONDecodeError) as exc:
			raise PersistenceError(
				f"Failed to load SQLite persistence database {self._path!r}: {exc}",
				backend="sqlite",
				path=self._path,
				) from exc

	def save_day(self, date: str, payload: dict) -> None:
		try:
			with self._lock, sqlite3.connect(self._path) as conn:
				conn.execute(
					"INSERT INTO daily_usage (date, payload) VALUES (?, ?) "
					"ON CONFLICT(date) DO UPDATE SET payload = excluded.payload",
					(date, json.dumps(payload)),
					)
		except (sqlite3.Error, TypeError) as exc:
			raise PersistenceError(
				f"Failed to save SQLite persistence day {date!r}: {exc}",
				backend="sqlite",
				path=self._path,
				) from exc

	def save_all(self, days: dict[str, dict]) -> None:
		try:
			with self._lock, sqlite3.connect(self._path) as conn:
				conn.executemany(
					"INSERT INTO daily_usage (date, payload) VALUES (?, ?) "
					"ON CONFLICT(date) DO UPDATE SET payload = excluded.payload",
					[(d, json.dumps(p)) for d, p in days.items()],
					)
		except (sqlite3.Error, TypeError) as exc:
			raise PersistenceError(
				f"Failed to save SQLite persistence database {self._path!r}: {exc}",
				backend="sqlite",
				path=self._path,
				) from exc
