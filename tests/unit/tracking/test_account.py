"""Unit tests for tracking/account_tracker.py."""

from __future__ import annotations

from poolgate.tracking.account import AccountTracker


class TestAccountTracker:

    def test_snapshot_all_empty_initially(self):
        tracker = AccountTracker()
        assert tracker.snapshot_all() == []

    def test_record_use_creates_entry(self):
        tracker = AccountTracker()
        tracker.record_use("key_1", tokens=100)
        result = tracker.snapshot_all()
        assert len(result) == 1
        assert result[0]["api_key"] == "key_1"

    def test_record_use_increments_today_counters(self):
        tracker = AccountTracker()
        tracker.record_use("key_1", tokens=50)
        tracker.record_use("key_1", tokens=30)
        snap = tracker.snapshot("key_1")
        assert snap["requests_today"] == 2
        assert snap["tokens_today"] == 80

    def test_record_use_zero_tokens_counts_request(self):
        tracker = AccountTracker()
        tracker.record_use("key_2", tokens=0)
        snap = tracker.snapshot("key_2")
        assert snap["requests_today"] == 1
        assert snap["tokens_today"] == 0

    def test_multiple_keys_tracked_independently(self):
        tracker = AccountTracker()
        tracker.record_use("key_a", tokens=10)
        tracker.record_use("key_b", tokens=20)
        snap_a = tracker.snapshot("key_a")
        snap_b = tracker.snapshot("key_b")
        assert snap_a["tokens_today"] == 10
        assert snap_b["tokens_today"] == 20

    def test_snapshot_unknown_key_returns_zeros(self):
        tracker = AccountTracker()
        snap = tracker.snapshot("unknown_key")
        assert snap["requests_today"] == 0
        assert snap["tokens_today"] == 0
        assert snap["last_used"] is None

    def test_last_used_is_set_after_record(self):
        tracker = AccountTracker()
        tracker.record_use("key_1", tokens=5)
        snap = tracker.snapshot("key_1")
        assert snap["last_used"] is not None

    def test_least_used_key_returns_none_for_empty_list(self):
        tracker = AccountTracker()
        assert tracker.least_used_key([]) is None

    def test_least_used_key_picks_lower_usage(self):
        tracker = AccountTracker()
        for _ in range(5):
            tracker.record_use("heavy_key", tokens=10)
        tracker.record_use("light_key", tokens=1)
        result = tracker.least_used_key(["heavy_key", "light_key"])
        assert result == "light_key"
