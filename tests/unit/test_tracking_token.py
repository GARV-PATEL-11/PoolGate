"""Unit tests for tracking/token_tracker.py."""

from __future__ import annotations

from tracking.token_tracker import TokenTracker


class TestTokenTracker:

    def test_snapshot_empty_initially(self):
        tracker = TokenTracker()
        assert tracker.snapshot() == {}

    def test_record_tracks_model(self):
        tracker = TokenTracker()
        tracker.record("llama-3.3-70b-versatile", tokens_in=10, tokens_out=5)
        snap = tracker.snapshot()
        assert "llama-3.3-70b-versatile" in snap

    def test_record_accumulates_tokens_in(self):
        tracker = TokenTracker()
        tracker.record("model-a", tokens_in=10, tokens_out=0)
        tracker.record("model-a", tokens_in=20, tokens_out=0)
        snap = tracker.snapshot()
        assert snap["model-a"]["tokens_in"] == 30

    def test_record_accumulates_tokens_out(self):
        tracker = TokenTracker()
        tracker.record("model-a", tokens_in=0, tokens_out=5)
        tracker.record("model-a", tokens_in=0, tokens_out=7)
        snap = tracker.snapshot()
        assert snap["model-a"]["tokens_out"] == 12

    def test_multiple_models_tracked_independently(self):
        tracker = TokenTracker()
        tracker.record("model-a", tokens_in=10, tokens_out=5)
        tracker.record("model-b", tokens_in=20, tokens_out=8)
        snap = tracker.snapshot()
        assert snap["model-a"]["tokens_in"] == 10
        assert snap["model-b"]["tokens_in"] == 20

    def test_tokens_for_day_returns_correct_structure(self):
        tracker = TokenTracker()
        tracker.record("model-x", tokens_in=15, tokens_out=7)
        day = tracker.tokens_for_day("model-x")
        assert "date" in day
        assert day["tokens_in"] == 15
        assert day["tokens_out"] == 7

    def test_tokens_for_day_unknown_model_returns_zeros(self):
        tracker = TokenTracker()
        day = tracker.tokens_for_day("unknown-model")
        assert day["tokens_in"] == 0
        assert day["tokens_out"] == 0


class TestTokenTrackerAliases:

    def test_get_session_usage_delegates_to_tokens_for_day(self):
        tracker = TokenTracker()
        tracker.record("model-x", tokens_in=20, tokens_out=10)
        result = tracker.get_session_usage("model-x")
        assert result["tokens_in"] == 20
        assert result["tokens_out"] == 10

    def test_get_key_usage_delegates_to_tokens_for_day(self):
        tracker = TokenTracker()
        tracker.record("model-x", tokens_in=5, tokens_out=3)
        result = tracker.get_key_usage("model-x")
        assert result["tokens_in"] == 5
        assert result["tokens_out"] == 3


class TestTokenTrackerRollingWindow:

    def test_tokens_in_last_minute_reflects_recorded_tokens(self):
        tracker = TokenTracker()
        tracker.record("model-x", tokens_in=100, tokens_out=50)
        total = tracker.tokens_in_last_minute("model-x")
        assert total == 150

    def test_tokens_in_last_24h_reflects_recorded_tokens(self):
        tracker = TokenTracker()
        tracker.record("model-x", tokens_in=200, tokens_out=80)
        total = tracker.tokens_in_last_24h("model-x")
        assert total == 280

    def test_tokens_in_last_minute_returns_zero_for_unknown_model(self):
        tracker = TokenTracker()
        assert tracker.tokens_in_last_minute("unknown") == 0

    def test_tokens_in_last_24h_returns_zero_for_unknown_model(self):
        tracker = TokenTracker()
        assert tracker.tokens_in_last_24h("unknown") == 0

    def test_remaining_tpm_is_limit_minus_used(self):
        tracker = TokenTracker()
        tracker.record("model-x", tokens_in=300, tokens_out=200)
        remaining = tracker.remaining_tpm("model-x", limit=1000)
        assert remaining == 500

    def test_remaining_tpm_clamps_to_zero(self):
        tracker = TokenTracker()
        tracker.record("model-x", tokens_in=900, tokens_out=200)
        remaining = tracker.remaining_tpm("model-x", limit=1000)
        assert remaining == 0

    def test_remaining_tpd_is_limit_minus_used(self):
        tracker = TokenTracker()
        tracker.record("model-x", tokens_in=1000, tokens_out=500)
        remaining = tracker.remaining_tpd("model-x", limit=10000)
        assert remaining == 8500

    def test_remaining_tpm_unknown_model_returns_limit(self):
        tracker = TokenTracker()
        assert tracker.remaining_tpm("unknown", limit=5000) == 5000


class TestTokenTrackerAllDays:

    def test_all_days_for_model_returns_empty_for_unknown_model(self):
        tracker = TokenTracker()
        assert tracker.all_days_for_model("unknown") == []

    def test_all_days_for_model_returns_sorted_history(self):
        tracker = TokenTracker()
        tracker.load_days(
            {
                "2026-06-02": {"model-x": {"tokens_in": 20, "tokens_out": 10}},
                "2026-06-01": {"model-x": {"tokens_in": 10, "tokens_out": 5}},
            }
        )
        days = tracker.all_days_for_model("model-x")
        assert len(days) == 2
        assert days[0]["date"] == "2026-06-01"
        assert days[1]["date"] == "2026-06-02"


class TestTokenTrackerPersistence:

    def test_load_days_and_export_days_round_trip(self):
        tracker = TokenTracker()
        original = {
            "2026-06-01": {
                "model-a": {"tokens_in": 100, "tokens_out": 50},
                "model-b": {"tokens_in": 200, "tokens_out": 80},
            },
            "2026-06-02": {
                "model-a": {"tokens_in": 150, "tokens_out": 70},
            },
        }
        tracker.load_days(original)
        exported = tracker.export_days()
        assert exported["2026-06-01"]["model-a"]["tokens_in"] == 100
        assert exported["2026-06-01"]["model-b"]["tokens_out"] == 80
        assert exported["2026-06-02"]["model-a"]["tokens_in"] == 150

    def test_load_days_handles_missing_tokens_in(self):
        tracker = TokenTracker()
        tracker.load_days({"2026-06-01": {"model-x": {"tokens_out": 42}}})
        day = tracker.tokens_for_day("model-x", "2026-06-01")
        assert day["tokens_in"] == 0
        assert day["tokens_out"] == 42

    def test_export_days_is_empty_when_no_records(self):
        tracker = TokenTracker()
        assert tracker.export_days() == {}
