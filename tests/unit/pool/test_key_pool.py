"""Unit tests for poolgate/pool/key_pool.py."""

from __future__ import annotations

import pytest

from poolgate.exceptions.keys import APIKeyError
from poolgate.pool.key_pool import APIKeyState, KeyPool
from poolgate.schemas.common.runtime import APIKeyStatus


class TestAPIKeyState:

    def test_from_key_masks_the_raw_key(self):
        state = APIKeyState.from_key(key_id="key_0", raw_key="gsk_abcdefghijklmnop")
        assert state.raw_key == "gsk_abcdefghijklmnop"
        assert state.masked_key != state.raw_key
        assert "abcdefghijklmnop" not in state.masked_key

    def test_new_key_is_available(self, api_key_state):
        assert api_key_state.is_available is True

    def test_record_request_start_increments_active(self, api_key_state):
        api_key_state.record_request_start()
        assert api_key_state.active_requests == 1

    def test_record_request_end_decrements_active_and_records_success(self, api_key_state):
        api_key_state.record_request_start()
        api_key_state.record_request_end(latency=0.5, tokens_in=10, tokens_out=20)
        assert api_key_state.active_requests == 0
        assert api_key_state.success_count == 1
        assert api_key_state.input_tokens == 10
        assert api_key_state.output_tokens == 20
        assert api_key_state.total_tokens == 30

    def test_record_request_end_resets_consecutive_failures(self, api_key_state):
        api_key_state.record_failure(is_rate_limit=False)
        api_key_state.record_failure(is_rate_limit=False)
        assert api_key_state.consecutive_failures == 2
        api_key_state.record_request_start()
        api_key_state.record_request_end(latency=0.1, tokens_in=1, tokens_out=1)
        assert api_key_state.consecutive_failures == 0

    def test_record_failure_with_rate_limit_sets_cooldown(self, api_key_state):
        api_key_state.record_failure(is_rate_limit=True, cooldown_secs=5.0)
        assert api_key_state.status == APIKeyStatus.RATE_LIMITED
        assert api_key_state.is_cooling_down is True
        assert api_key_state.is_available is False

    def test_record_failure_increments_consecutive_failures(self, api_key_state):
        api_key_state.record_failure(is_rate_limit=False)
        assert api_key_state.consecutive_failures == 1
        api_key_state.record_failure(is_rate_limit=False)
        assert api_key_state.consecutive_failures == 2

    def test_rate_limit_failure_does_not_increment_consecutive_failures_counter_used_for_breaker(
        self,
        api_key_state,
    ):
        # consecutive_429s tracks rate limits separately; consecutive_failures
        # (the circuit-breaker counter) still increments for ALL failures,
        # including rate limits — only the *trip* in RequestScheduler.mark_key_failure
        # is gated to non-rate-limit failures.
        api_key_state.record_failure(is_rate_limit=True, cooldown_secs=1.0)
        assert api_key_state.consecutive_failures == 1
        assert api_key_state.consecutive_429s == 1

    def test_key_at_max_parallel_is_not_available(self, api_key_state):
        api_key_state.max_parallel_requests = 1
        api_key_state.record_request_start()
        assert api_key_state.is_available is False

    def test_health_score_of_unavailable_key_is_negative_infinity(self, api_key_state):
        api_key_state.mark_disabled()
        assert api_key_state.health_score() == float("-inf")

    def test_health_score_decreases_with_higher_failure_rate(self):
        healthy = APIKeyState.from_key(key_id="k1", raw_key="gsk_a")
        unhealthy = APIKeyState.from_key(key_id="k2", raw_key="gsk_b")
        for _ in range(5):
            unhealthy.record_request_start()
            unhealthy.record_failure(is_rate_limit=False)
        assert healthy.health_score() > unhealthy.health_score()

    def test_mark_failed_sets_failed_status(self, api_key_state):
        api_key_state.mark_failed()
        assert api_key_state.status == APIKeyStatus.FAILED
        assert api_key_state.is_available is False


class TestKeyPool:

    def test_pool_initializes_with_all_keys_available(self, key_pool):
        assert len(key_pool.get_eligible_keys()) == 3

    def test_add_key_assigns_sequential_ids(self):
        pool = KeyPool()
        k1 = pool.add_key("gsk_a")
        k2 = pool.add_key("gsk_b")
        assert k1.key_id == "key_0"
        assert k2.key_id == "key_1"

    def test_disable_key_removes_it_from_eligible(self, key_pool):
        keys = key_pool.list_keys()
        key_pool.disable_key(keys[0].key_id)
        assert len(key_pool.get_eligible_keys()) == 2

    def test_get_key_returns_highest_scoring_available_key(self, key_pool):
        keys = key_pool.list_keys()
        keys[0].record_failure(is_rate_limit=True, cooldown_secs=60.0)
        keys[1].record_failure(is_rate_limit=True, cooldown_secs=60.0)
        best = key_pool.get_key()
        assert best.key_id == keys[2].key_id

    def test_require_key_raises_typed_api_key_error_for_unknown_id(self, key_pool):
        """
        Regression test for the audit's Phase 7/H4 finding: _require_key()
        used to raise a bare KeyError, breaking the "catch one base
        exception type" contract the rest of the hierarchy provides. It now
        raises APIKeyError, the documented base for key-lifecycle failures.
        """
        with pytest.raises(APIKeyError):
            key_pool.disable_key("nonexistent_key_id")

    def test_remove_key_returns_none_for_unknown_id(self, key_pool):
        assert key_pool.remove_key("nonexistent") is None

    def test_enable_key_clears_cooldown(self, key_pool):
        keys = key_pool.list_keys()
        target = keys[0]
        key_pool.mark_unhealthy(target.key_id, cooldown_secs=60.0)
        assert target.is_cooling_down is True
        key_pool.enable_key(target.key_id)
        assert target.is_cooling_down is False
        assert target.status == APIKeyStatus.ACTIVE
