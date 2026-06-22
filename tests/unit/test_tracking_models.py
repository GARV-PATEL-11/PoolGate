"""Unit tests for tracking/models.py — shared data types."""

from __future__ import annotations

from datetime import datetime, timezone

from tracking.models import DailyBucket, TokenStats, TokenUsage, today_str


class TestTokenUsage:
    def test_initial_values_are_zero(self):
        u = TokenUsage()
        assert u.tokens_in == 0
        assert u.tokens_out == 0

    def test_total_property_sums_in_and_out(self):
        u = TokenUsage(tokens_in=10, tokens_out=5)
        assert u.total == 15

    def test_add_accumulates_tokens(self):
        u = TokenUsage()
        u.add(tokens_in=5, tokens_out=3)
        assert u.tokens_in == 5
        assert u.tokens_out == 3

    def test_add_is_cumulative(self):
        u = TokenUsage(tokens_in=10, tokens_out=2)
        u.add(tokens_in=5, tokens_out=3)
        assert u.tokens_in == 15
        assert u.tokens_out == 5

    def test_total_after_add(self):
        u = TokenUsage()
        u.add(tokens_in=7, tokens_out=3)
        assert u.total == 10


class TestDailyBucket:
    def test_to_dict_contains_all_keys(self):
        bucket = DailyBucket(
            date="2026-06-01",
            requests=5,
            successful_requests=4,
            failed_requests=1,
            tokens_in=100,
            tokens_out=50,
        )
        d = bucket.to_dict()
        assert d["date"] == "2026-06-01"
        assert d["requests"] == 5
        assert d["successful_requests"] == 4
        assert d["failed_requests"] == 1
        assert d["tokens_in"] == 100
        assert d["tokens_out"] == 50

    def test_from_dict_round_trips(self):
        original = DailyBucket(
            date="2026-06-01",
            requests=10,
            successful_requests=9,
            failed_requests=1,
            tokens_in=200,
            tokens_out=80,
        )
        restored = DailyBucket.from_dict(original.to_dict())
        assert restored.date == original.date
        assert restored.requests == original.requests
        assert restored.successful_requests == original.successful_requests
        assert restored.tokens_in == original.tokens_in
        assert restored.tokens_out == original.tokens_out

    def test_from_dict_handles_missing_fields(self):
        bucket = DailyBucket.from_dict({"date": "2026-06-01"})
        assert bucket.requests == 0
        assert bucket.tokens_in == 0

    def test_total_tokens_property(self):
        bucket = DailyBucket(date="2026-06-01", tokens_in=300, tokens_out=120)
        assert bucket.total_tokens == 420


class TestTodayStr:
    def test_returns_iso_date_string(self):
        result = today_str()
        expected = datetime.now(timezone.utc).date().isoformat()
        assert result == expected

    def test_format_is_yyyy_mm_dd(self):
        result = today_str()
        parts = result.split("-")
        assert len(parts) == 3
        assert len(parts[0]) == 4
        assert len(parts[1]) == 2
        assert len(parts[2]) == 2


class TestTokenStats:
    def test_total_tokens_property(self):
        stats = TokenStats(scope="test-model", tokens_in=50, tokens_out=30)
        assert stats.total_tokens == 80

    def test_zero_totals(self):
        stats = TokenStats(scope="empty")
        assert stats.total_tokens == 0
