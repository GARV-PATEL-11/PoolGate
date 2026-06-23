"""Unit tests for tracking/request_tracker.py."""

from __future__ import annotations

from tracking.request_tracker import RequestTracker


class TestRequestTracker:

	def test_fresh_tracker_has_zero_rpm(self):
		tracker = RequestTracker()
		assert tracker.requests_per_minute("global") == 0

	def test_record_request_increments_count(self):
		tracker = RequestTracker()
		tracker.record_request("model-a")
		assert tracker.requests_per_minute("model-a") >= 1

	def test_record_request_default_scope_is_global(self):
		tracker = RequestTracker()
		tracker.record_request()
		assert tracker.requests_per_minute("global") == 1

	def test_multiple_requests_accumulate(self):
		tracker = RequestTracker()
		for _ in range(5):
			tracker.record_request("scope-x")
		assert tracker.requests_per_minute("scope-x") == 5

	def test_different_scopes_tracked_independently(self):
		tracker = RequestTracker()
		tracker.record_request("scope-a")
		tracker.record_request("scope-a")
		tracker.record_request("scope-b")
		assert tracker.requests_per_minute("scope-a") == 2
		assert tracker.requests_per_minute("scope-b") == 1

	def test_snapshot_returns_scope_and_counts(self):
		tracker = RequestTracker()
		tracker.record_request("my-model")
		snap = tracker.snapshot("my-model")
		assert snap["scope"] == "my-model"
		assert "requests_per_minute" in snap
		assert "requests_per_day" in snap

	def test_unknown_scope_returns_zero(self):
		tracker = RequestTracker()
		assert tracker.requests_per_minute("nonexistent") == 0
		assert tracker.requests_per_day("nonexistent") == 0
