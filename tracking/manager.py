"""
tracking.manager
----------------
Small process-local coordinator for the tracking subsystem.

This keeps the request/usage/token/account/quota trackers in one place so the
service layer can update them consistently without knowing about each tracker
individually.
"""

from __future__ import annotations

from typing import Any

from tracking.account_tracker import AccountTracker
from tracking.quota_tracker import QuotaTracker
from tracking.request_tracker import RequestTracker
from tracking.token_tracker import TokenTracker
from tracking.usage_tracker import UsageTracker


class TrackingManager:
    """Bundle of trackers with a small update API for the service layer."""

    def __init__(self) -> None:
        self.usage_tracker = UsageTracker()
        self.request_tracker = RequestTracker()
        self.token_tracker = TokenTracker()
        self.quota_tracker = QuotaTracker()
        self.account_tracker = AccountTracker()

    def record_success(
        self,
        model: str,
        *,
        tokens_in: int,
        tokens_out: int,
        api_key_id: str | None = None,
        scope: str | None = None,
        retried: bool = False,
    ) -> None:
        request_scope = scope or model
        self.usage_tracker.record_success(tokens_in, tokens_out, retried=retried)
        self.request_tracker.record_request(request_scope)
        self.token_tracker.record(model, tokens_in=tokens_in, tokens_out=tokens_out)
        if api_key_id:
            self.account_tracker.record_use(api_key_id, tokens_in + tokens_out)

    def record_failure(
        self,
        model: str,
        *,
        api_key_id: str | None = None,
        scope: str | None = None,
        retried: bool = False,
    ) -> None:
        request_scope = scope or model
        self.usage_tracker.record_failure(retried=retried)
        self.request_tracker.record_request(request_scope)
        if api_key_id:
            self.account_tracker.record_use(api_key_id, 0)

    def update_quota_from_headers(self, model: str, headers: dict[str, Any]) -> None:
        self.quota_tracker.update_from_headers(model, headers)

    def snapshot(self) -> dict[str, Any]:
        return {
            "usage": self.usage_tracker.snapshot(),
            "request_scopes": self.request_tracker.snapshot(),
            "token_usage": self.token_tracker.snapshot(),
            "quota": self.quota_tracker.snapshot_all(),
            "account_usage": self.account_tracker.snapshot_all(),
        }
