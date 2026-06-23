"""Unit tests for tracking/usage_tracker.py."""

from __future__ import annotations

from tracking.usage_tracker import UsageTracker


class TestUsageTracker:

    def test_snapshot_keys_present_on_fresh_tracker(self):
        tracker = UsageTracker()
        snap = tracker.snapshot()
        expected_keys = {
            "total_requests",
            "successful_requests",
            "failed_requests",
            "total_retries",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "success_rate",
        }
        assert set(snap.keys()) == expected_keys

    def test_success_rate_is_one_with_no_requests(self):
        tracker = UsageTracker()
        assert tracker.snapshot()["success_rate"] == 1.0

    def test_record_success_increments_counters(self):
        tracker = UsageTracker()
        tracker.record_success(tokens_in=10, tokens_out=5)
        snap = tracker.snapshot()
        assert snap["total_requests"] == 1
        assert snap["successful_requests"] == 1
        assert snap["failed_requests"] == 0
        assert snap["input_tokens"] == 10
        assert snap["output_tokens"] == 5
        assert snap["total_tokens"] == 15

    def test_record_failure_increments_failed_requests(self):
        tracker = UsageTracker()
        tracker.record_failure()
        snap = tracker.snapshot()
        assert snap["total_requests"] == 1
        assert snap["failed_requests"] == 1
        assert snap["successful_requests"] == 0

    def test_multiple_successes_accumulate_tokens(self):
        tracker = UsageTracker()
        tracker.record_success(tokens_in=10, tokens_out=5)
        tracker.record_success(tokens_in=20, tokens_out=8)
        snap = tracker.snapshot()
        assert snap["input_tokens"] == 30
        assert snap["output_tokens"] == 13
        assert snap["total_tokens"] == 43

    def test_success_rate_with_mixed_results(self):
        tracker = UsageTracker()
        tracker.record_success(tokens_in=1, tokens_out=1)
        tracker.record_success(tokens_in=1, tokens_out=1)
        tracker.record_failure()
        snap = tracker.snapshot()
        assert snap["total_requests"] == 3
        assert snap["success_rate"] == round(2 / 3, 4)

    def test_retried_true_increments_total_retries(self):
        tracker = UsageTracker()
        tracker.record_success(tokens_in=5, tokens_out=2, retried=True)
        snap = tracker.snapshot()
        assert snap["total_retries"] == 1

    def test_retried_false_does_not_increment_retries(self):
        tracker = UsageTracker()
        tracker.record_success(tokens_in=5, tokens_out=2, retried=False)
        snap = tracker.snapshot()
        assert snap["total_retries"] == 0

    def test_export_days_returns_daily_history(self):
        tracker = UsageTracker()
        tracker.record_success(tokens_in=10, tokens_out=5)
        days = tracker.export_days()
        assert len(days) == 1

    def test_load_days_restores_history(self):
        tracker = UsageTracker()
        tracker.load_days(
            {
                "2026-06-01": {
                    "date": "2026-06-01",
                    "requests": 10,
                    "successful_requests": 9,
                    "failed_requests": 1,
                    "tokens_in": 100,
                    "tokens_out": 50,
                },
            },
        )
        days = tracker.export_days()
        assert "2026-06-01" in days
