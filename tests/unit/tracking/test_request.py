"""Unit tests for tracking/request_tracker.py."""

from __future__ import annotations

from poolgate.tracking.request import RequestTracker


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


class TestRequestTrackerAliases:

    def test_record_alias_increments_count(self):
        tracker = RequestTracker()
        tracker.record("scope-a")
        assert tracker.requests_per_minute("scope-a") == 1

    def test_update_alias_increments_count(self):
        tracker = RequestTracker()
        tracker.update("scope-b")
        assert tracker.requests_per_minute("scope-b") == 1

    def test_get_request_returns_snapshot_dict(self):
        tracker = RequestTracker()
        tracker.record_request("my-scope")
        result = tracker.get_request("my-scope")
        assert isinstance(result, dict)
        assert result["scope"] == "my-scope"
        assert result["requests_per_minute"] == 1


class TestRequestTrackerRemaining:

    def test_remaining_rpm_is_limit_minus_used(self):
        tracker = RequestTracker()
        for _ in range(3):
            tracker.record_request("key_0")
        remaining = tracker.remaining_rpm("key_0", limit=10)
        assert remaining == 7

    def test_remaining_rpm_clamps_to_zero(self):
        tracker = RequestTracker()
        for _ in range(15):
            tracker.record_request("key_0")
        remaining = tracker.remaining_rpm("key_0", limit=10)
        assert remaining == 0

    def test_remaining_rpd_is_limit_minus_used(self):
        tracker = RequestTracker()
        for _ in range(5):
            tracker.record_request("key_0")
        remaining = tracker.remaining_rpd("key_0", limit=100)
        assert remaining == 95

    def test_remaining_rpm_unknown_scope_returns_limit(self):
        tracker = RequestTracker()
        assert tracker.remaining_rpm("unknown", limit=30) == 30

    def test_remaining_rpd_unknown_scope_returns_limit(self):
        tracker = RequestTracker()
        assert tracker.remaining_rpd("unknown", limit=1000) == 1000


class TestRequestTrackerSecondsUntilFree:

    def test_seconds_until_rpm_frees_up_returns_zero_for_unknown_scope(self):
        tracker = RequestTracker()
        assert tracker.seconds_until_rpm_frees_up("nonexistent") == 0.0

    def test_seconds_until_rpd_frees_up_returns_zero_for_unknown_scope(self):
        tracker = RequestTracker()
        assert tracker.seconds_until_rpd_frees_up("nonexistent") == 0.0

    def test_seconds_until_rpm_frees_up_returns_float_for_known_scope(self):
        tracker = RequestTracker()
        tracker.record_request("key_0")
        result = tracker.seconds_until_rpm_frees_up("key_0")
        assert isinstance(result, float)

    def test_seconds_until_rpd_frees_up_returns_float_for_known_scope(self):
        tracker = RequestTracker()
        tracker.record_request("key_0")
        result = tracker.seconds_until_rpd_frees_up("key_0")
        assert isinstance(result, float)
