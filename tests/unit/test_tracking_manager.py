"""Unit tests for tracking/manager.py — TrackingManager."""

from __future__ import annotations

from tracking.manager import TrackingManager


class TestTrackingManager:

    def test_record_success_updates_usage_tracker(self):
        mgr = TrackingManager()
        mgr.record_success("llama-3.3-70b-versatile", tokens_in=10, tokens_out=5)
        snap = mgr.usage_tracker.snapshot()
        assert snap["successful_requests"] == 1
        assert snap["input_tokens"] == 10

    def test_record_success_updates_request_tracker(self):
        mgr = TrackingManager()
        mgr.record_success("llama-3.3-70b-versatile", tokens_in=10, tokens_out=5)
        assert mgr.request_tracker.requests_per_minute("llama-3.3-70b-versatile") == 1

    def test_record_success_updates_token_tracker(self):
        mgr = TrackingManager()
        mgr.record_success("llama-3.3-70b-versatile", tokens_in=10, tokens_out=5)
        snap = mgr.token_tracker.snapshot()
        assert "llama-3.3-70b-versatile" in snap
        assert snap["llama-3.3-70b-versatile"]["tokens_in"] == 10

    def test_record_success_with_api_key_id_updates_account_tracker(self):
        mgr = TrackingManager()
        mgr.record_success("model-x", tokens_in=10, tokens_out=5, api_key_id="key_1")
        accounts = mgr.account_tracker.snapshot_all()
        assert any(a["api_key"] == "key_1" for a in accounts)

    def test_record_success_without_api_key_id_skips_account_tracker(self):
        mgr = TrackingManager()
        mgr.record_success("model-x", tokens_in=10, tokens_out=5)
        assert mgr.account_tracker.snapshot_all() == []

    def test_record_failure_updates_usage_tracker(self):
        mgr = TrackingManager()
        mgr.record_failure("model-x")
        snap = mgr.usage_tracker.snapshot()
        assert snap["failed_requests"] == 1

    def test_record_failure_updates_request_tracker(self):
        mgr = TrackingManager()
        mgr.record_failure("model-x")
        assert mgr.request_tracker.requests_per_minute("model-x") == 1

    def test_record_failure_with_api_key_id_updates_account_tracker(self):
        mgr = TrackingManager()
        mgr.record_failure("model-x", api_key_id="key_2")
        accounts = mgr.account_tracker.snapshot_all()
        assert any(a["api_key"] == "key_2" for a in accounts)

    def test_snapshot_contains_all_sections(self):
        mgr = TrackingManager()
        snap = mgr.snapshot()
        assert "usage" in snap
        assert "request_scopes" in snap
        assert "token_usage" in snap
        assert "quota" in snap
        assert "account_usage" in snap

    def test_multiple_calls_accumulate(self):
        mgr = TrackingManager()
        mgr.record_success("model-x", tokens_in=10, tokens_out=5)
        mgr.record_success("model-x", tokens_in=20, tokens_out=8)
        snap = mgr.usage_tracker.snapshot()
        assert snap["successful_requests"] == 2
        assert snap["input_tokens"] == 30
