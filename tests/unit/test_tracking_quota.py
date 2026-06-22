"""Unit tests for tracking/quota_tracker.py."""

from __future__ import annotations

from tracking.quota_tracker import QuotaTracker, _parse_duration


class TestQuotaTracker:
    def test_snapshot_all_empty_initially(self):
        tracker = QuotaTracker()
        assert tracker.snapshot_all() == []

    def test_update_stores_quota_for_model(self):
        tracker = QuotaTracker()
        tracker.update("llama-3.3-70b-versatile", remaining_rpd=900, remaining_tpd=50000)
        snap = tracker.get("llama-3.3-70b-versatile")
        assert snap is not None
        assert snap["remaining_rpd"] == 900
        assert snap["remaining_tpd"] == 50000

    def test_update_from_headers_parses_groq_headers(self):
        tracker = QuotaTracker()
        headers = {
            "x-ratelimit-remaining-requests": "950",
            "x-ratelimit-remaining-tokens": "45000",
        }
        tracker.update_from_headers("llama-3.3-70b-versatile", headers)
        snap = tracker.get("llama-3.3-70b-versatile")
        assert snap["remaining_rpd"] == 950
        assert snap["remaining_tpd"] == 45000

    def test_update_from_headers_handles_missing_headers(self):
        tracker = QuotaTracker()
        tracker.update_from_headers("llama-3.3-70b-versatile", {})
        snap = tracker.get("llama-3.3-70b-versatile")
        assert snap["remaining_rpd"] is None
        assert snap["remaining_tpd"] is None

    def test_update_from_headers_parses_reset_seconds_as_float(self):
        tracker = QuotaTracker()
        headers = {"x-ratelimit-reset-requests": "45.5"}
        tracker.update_from_headers("model-x", headers)
        snap = tracker.get("model-x")
        assert snap["reset_requests_seconds"] == 45.5

    def test_update_from_headers_parses_duration_string(self):
        tracker = QuotaTracker()
        headers = {"x-ratelimit-reset-requests": "2m30s"}
        tracker.update_from_headers("model-x", headers)
        snap = tracker.get("model-x")
        assert snap["reset_requests_seconds"] == 150.0

    def test_snapshot_all_returns_all_models(self):
        tracker = QuotaTracker()
        tracker.update("model-a", remaining_rpd=100)
        tracker.update("model-b", remaining_rpd=200)
        all_snaps = tracker.snapshot_all()
        assert len(all_snaps) == 2

    def test_is_exhausted_false_when_no_snapshot(self):
        tracker = QuotaTracker()
        assert tracker.is_exhausted("unknown-model") is False

    def test_is_exhausted_true_when_remaining_rpd_zero(self):
        tracker = QuotaTracker()
        tracker.update("model-x", remaining_rpd=0)
        assert tracker.is_exhausted("model-x") is True

    def test_is_exhausted_false_when_remaining_rpd_positive(self):
        tracker = QuotaTracker()
        tracker.update("model-x", remaining_rpd=10)
        assert tracker.is_exhausted("model-x") is False

    def test_get_returns_none_for_unknown_model(self):
        tracker = QuotaTracker()
        assert tracker.get("not-registered") is None


class TestParseDuration:
    def test_parses_minutes_and_seconds(self):
        assert _parse_duration("2m30s") == 150.0

    def test_parses_seconds_only(self):
        assert _parse_duration("45.5s") == 45.5

    def test_parses_minutes_only(self):
        assert _parse_duration("3m") == 180.0

    def test_returns_none_for_invalid_string(self):
        assert _parse_duration("invalid") is None
