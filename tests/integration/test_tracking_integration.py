"""
Integration tests for TrackingManager + PersistenceService.

Verifies that usage data written by TrackingManager survives a flush/reload
cycle through both JSON and SQLite backends — no mocking of persistence layers.

Note: UsageTracker.snapshot() returns lifetime totals (since process start).
After load_days(), use snapshot_for_day() to read today's persisted bucket.
"""

from __future__ import annotations

from services.persistence_service import PersistenceService
from tracking.manager import TrackingManager


def _record_calls(manager: TrackingManager, n: int = 3) -> None:
	for i in range(n):
		manager.record_success(
			"llama-3.3-70b-versatile",
			tokens_in=10 + i,
			tokens_out=5 + i,
			api_key_id=f"key_{i % 2}",
			)


class TestTrackingManagerWithJSONBackend:

	def test_flush_then_reload_preserves_request_count(self, tmp_path):
		path = str(tmp_path / "usage.json")
		persistence = PersistenceService.json(path)

		manager = TrackingManager()
		_record_calls(manager, n=4)
		persistence.flush_tracker(manager.usage_tracker)

		manager2 = TrackingManager()
		persistence.load_tracker(manager2.usage_tracker)
		# snapshot_for_day() reads from the reloaded _days, not _lifetime
		snap = manager2.usage_tracker.snapshot_for_day()
		assert snap["requests"] >= 4
		assert snap["successful_requests"] >= 4

	def test_flush_then_reload_preserves_token_counts(self, tmp_path):
		path = str(tmp_path / "tokens.json")
		persistence = PersistenceService.json(path)

		manager = TrackingManager()
		manager.record_success(
			"llama-3.3-70b-versatile",
			tokens_in=100,
			tokens_out=50,
			api_key_id="k0",
			)
		persistence.flush_tracker(manager.token_tracker)

		manager2 = TrackingManager()
		persistence.load_tracker(manager2.token_tracker)
		# TokenTracker snapshot — per-model breakdown: {model: {"tokens_in": X, ...}}
		snap = manager2.token_tracker.snapshot()
		model_data = snap.get("llama-3.3-70b-versatile", {})
		assert model_data.get("tokens_in", 0) >= 100
		assert model_data.get("tokens_out", 0) >= 50

	def test_multiple_flushes_accumulate(self, tmp_path):
		path = str(tmp_path / "multi.json")
		persistence = PersistenceService.json(path)

		manager1 = TrackingManager()
		_record_calls(manager1, n=2)
		persistence.flush_tracker(manager1.usage_tracker)

		manager2 = TrackingManager()
		persistence.load_tracker(manager2.usage_tracker)
		_record_calls(manager2, n=3)
		persistence.flush_tracker(manager2.usage_tracker)

		manager3 = TrackingManager()
		persistence.load_tracker(manager3.usage_tracker)
		snap = manager3.usage_tracker.snapshot_for_day()
		assert snap["requests"] >= 5

	def test_failure_records_survive_reload(self, tmp_path):
		path = str(tmp_path / "failures.json")
		persistence = PersistenceService.json(path)

		manager = TrackingManager()
		manager.record_success(
			"llama-3.3-70b-versatile",
			tokens_in=5,
			tokens_out=2,
			api_key_id="k0",
			)
		manager.record_failure(
			"llama-3.3-70b-versatile",
			api_key_id="k0",
			)
		persistence.flush_tracker(manager.usage_tracker)

		manager2 = TrackingManager()
		persistence.load_tracker(manager2.usage_tracker)
		snap = manager2.usage_tracker.snapshot_for_day()
		assert snap["requests"] >= 2
		assert snap["failed_requests"] >= 1

	def test_load_from_empty_file_is_safe(self, tmp_path):
		path = str(tmp_path / "empty.json")
		persistence = PersistenceService.json(path)

		manager = TrackingManager()
		persistence.load_tracker(manager.usage_tracker)
		snap = manager.usage_tracker.snapshot_for_day()
		assert snap["requests"] == 0


class TestTrackingManagerWithSQLiteBackend:

	def test_sqlite_flush_then_reload_preserves_request_count(self, tmp_path):
		db_path = str(tmp_path / "usage.db")
		persistence = PersistenceService.sqlite(db_path)

		manager = TrackingManager()
		_record_calls(manager, n=5)
		persistence.flush_tracker(manager.usage_tracker)

		manager2 = TrackingManager()
		persistence.load_tracker(manager2.usage_tracker)
		snap = manager2.usage_tracker.snapshot_for_day()
		assert snap["requests"] >= 5

	def test_sqlite_load_all_returns_dict(self, tmp_path):
		db_path = str(tmp_path / "load_test.db")
		persistence = PersistenceService.sqlite(db_path)

		manager = TrackingManager()
		_record_calls(manager, n=1)
		persistence.flush_tracker(manager.usage_tracker)

		data = persistence.load_all()
		assert isinstance(data, dict)
		assert len(data) >= 1

	def test_sqlite_and_json_same_round_trip_semantics(self, tmp_path):
		json_persistence = PersistenceService.json(str(tmp_path / "u.json"))
		sqlite_persistence = PersistenceService.sqlite(str(tmp_path / "u.db"))

		for p in [json_persistence, sqlite_persistence]:
			mgr = TrackingManager()
			_record_calls(mgr, n=3)
			p.flush_tracker(mgr.usage_tracker)

			mgr2 = TrackingManager()
			p.load_tracker(mgr2.usage_tracker)
			snap = mgr2.usage_tracker.snapshot_for_day()
			assert snap["requests"] >= 3, f"Round-trip failed for {p!r}"


class TestMultiTrackerFlush:

	def test_all_three_trackers_survive_reload(self, tmp_path):
		"""Each tracker uses its own persistence file (as the module docs require)."""
		usage_p = PersistenceService.json(str(tmp_path / "usage.json"))
		token_p = PersistenceService.json(str(tmp_path / "tokens.json"))
		account_p = PersistenceService.json(str(tmp_path / "account.json"))

		manager = TrackingManager()
		manager.record_success(
			"llama-3.3-70b-versatile",
			tokens_in=20,
			tokens_out=10,
			api_key_id="k0",
			)

		usage_p.flush_tracker(manager.usage_tracker)
		token_p.flush_tracker(manager.token_tracker)
		account_p.flush_tracker(manager.account_tracker)

		manager2 = TrackingManager()
		usage_p.load_tracker(manager2.usage_tracker)
		token_p.load_tracker(manager2.token_tracker)
		account_p.load_tracker(manager2.account_tracker)

		usage_snap = manager2.usage_tracker.snapshot_for_day()
		token_snap = manager2.token_tracker.snapshot()
		model_data = token_snap.get("llama-3.3-70b-versatile", {})

		assert usage_snap["requests"] >= 1
		assert model_data.get("tokens_in", 0) >= 20
		assert model_data.get("tokens_out", 0) >= 10
