"""Unit tests for services/session_service.py."""

from __future__ import annotations

import time

from services.session_service import SessionManager, SessionUsageTracker


class TestSessionUsageTracker:

	def _tracker(self, ttl: float = 60.0) -> SessionUsageTracker:
		t = SessionUsageTracker(session_id="sess-1")
		t.expires_at = time.time() + ttl
		return t

	def test_record_success_increments_counters(self):
		t = self._tracker()
		t.record_success(model="llama-3.3-70b-versatile", tokens_in=10, tokens_out=5, latency=0.3)
		assert t.total_requests == 1
		assert t.successful_requests == 1
		assert t.input_tokens == 10
		assert t.output_tokens == 5

	def test_record_failure_increments_counters(self):
		t = self._tracker()
		t.record_failure()
		assert t.total_requests == 1
		assert t.failed_requests == 1

	def test_total_tokens_property(self):
		t = self._tracker()
		t.record_success(model="m", tokens_in=7, tokens_out=3, latency=0.1)
		assert t.total_tokens == 10

	def test_model_usage_tracks_per_model(self):
		t = self._tracker()
		t.record_success(model="llama", tokens_in=5, tokens_out=2, latency=0.2)
		t.record_success(model="llama", tokens_in=3, tokens_out=1, latency=0.1)
		usage = t.model_usage
		assert "llama" in usage
		assert usage["llama"]["request_count"] == 2

	def test_is_expired_false_for_future_expiry(self):
		t = self._tracker(ttl=60.0)
		assert t.is_expired is False

	def test_is_expired_true_for_past_expiry(self):
		t = SessionUsageTracker(session_id="x")
		t.expires_at = time.time() - 1.0
		assert t.is_expired is True


class TestSessionManager:

	def test_get_or_create_creates_new_session(self):
		manager = SessionManager(session_ttl_hours=1)
		tracker = manager.get_or_create("sess-1")
		assert tracker.session_id == "sess-1"

	def test_get_or_create_returns_existing_session(self):
		manager = SessionManager(session_ttl_hours=1)
		t1 = manager.get_or_create("sess-2")
		t2 = manager.get_or_create("sess-2")
		assert t1 is t2

	def test_get_returns_none_for_unknown_session(self):
		manager = SessionManager(session_ttl_hours=1)
		assert manager.get("nonexistent") is None

	def test_get_returns_none_for_expired_session(self):
		manager = SessionManager(session_ttl_hours=1)
		tracker = manager.get_or_create("sess-3")
		tracker.expires_at = time.time() - 1.0
		assert manager.get("sess-3") is None

	def test_active_count_increments_on_creation(self):
		manager = SessionManager(session_ttl_hours=1)
		manager.get_or_create("sess-a")
		manager.get_or_create("sess-b")
		assert manager.active_count() >= 2

	def test_expire_old_sessions_removes_expired(self):
		manager = SessionManager(session_ttl_hours=1)
		tracker = manager.get_or_create("sess-expire")
		tracker.expires_at = time.time() - 1.0
		removed = manager.expire_old_sessions()
		assert removed >= 1
		assert manager.get("sess-expire") is None
