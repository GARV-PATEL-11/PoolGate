"""Service-level persistence facade for tracker history."""

from __future__ import annotations

from typing import Protocol

from tracking.persistence import JSONPersistence, Persistence, SQLitePersistence


class PersistableTracker(Protocol):

	def export_days(self) -> dict[str, dict]: ...

	def load_days(self, days: dict[str, dict]) -> None: ...


class PersistenceService:
	"""Coordinates loading and flushing daily tracker state."""

	def __init__(self, backend: Persistence | None = None) -> None:
		self._backend = backend or JSONPersistence()

	@classmethod
	def json(cls, path: str) -> PersistenceService:
		return cls(JSONPersistence(path))

	@classmethod
	def sqlite(cls, path: str) -> PersistenceService:
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
