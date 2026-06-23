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
