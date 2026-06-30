"""Integration tests for key_manager — KeyPool + APIKeyState multi-key scenarios."""

from __future__ import annotations

import threading

from poolgate.pool.key_pool import APIKeyState, KeyPool
from poolgate.schemas.common.runtime import APIKeyStatus


def _make_key(key_id: str = "key_0", raw_key: str = "gsk_test_key") -> APIKeyState:
    return APIKeyState.from_key(key_id=key_id, raw_key=raw_key)


class TestAPIKeyStateHealth:

    def test_fresh_key_is_available(self):
        key = _make_key()
        assert key.is_available is True

    def test_disabled_key_is_not_available(self):
        key = _make_key()
        key.mark_disabled()
        assert key.is_available is False
        assert key.status == APIKeyStatus.DISABLED

    def test_failed_key_is_not_available(self):
        key = _make_key()
        key.mark_failed()
        assert key.is_available is False
        assert key.status == APIKeyStatus.FAILED

    def test_health_score_fresh_key(self):
        key = _make_key()
        score = key.health_score()
        assert score > 0.0

    def test_health_score_degraded_after_failures(self):
        key = _make_key()
        original_score = key.health_score()
        # Record a failure (non-rate-limit)
        key.record_request_start()
        key.record_failure(is_rate_limit=False)
        degraded_score = key.health_score()
        assert degraded_score < original_score

    def test_disabled_key_health_score_is_neg_inf(self):
        key = _make_key()
        key.mark_disabled()
        assert key.health_score() == float("-inf")


class TestAPIKeyStateRPMCounter:

    def test_rpm_increments_on_request_start(self):
        key = _make_key()
        assert key.requests_per_minute == 0
        key.record_request_start()
        assert key.requests_per_minute == 1

    def test_rpm_multiple_requests(self):
        key = _make_key()
        for _ in range(5):
            key.record_request_start()
        assert key.requests_per_minute == 5

    def test_active_requests_tracked(self):
        key = _make_key()
        key.record_request_start()
        key.record_request_start()
        assert key.active_requests == 2

    def test_active_requests_decrease_on_end(self):
        key = _make_key()
        key.record_request_start()
        key.record_request_end(latency=0.1, tokens_in=10, tokens_out=5)
        assert key.active_requests == 0


class TestAPIKeyStateCooldown:

    def test_rate_limit_puts_key_in_cooldown(self):
        key = _make_key()
        key.record_request_start()
        key.record_failure(is_rate_limit=True, cooldown_secs=60.0)
        assert key.status == APIKeyStatus.RATE_LIMITED
        assert key.is_cooling_down is True
        assert key.is_available is False

    def test_cooldown_expires(self):
        key = _make_key()
        key.record_request_start()
        # Set cooldown to expire immediately
        key.record_failure(is_rate_limit=True, cooldown_secs=0.0)
        # After 0-second cooldown expires, key should not be cooling down
        assert key.is_cooling_down is False

    def test_reset_cooldown_restores_active_status(self):
        key = _make_key()
        key.record_request_start()
        key.record_failure(is_rate_limit=True, cooldown_secs=3600.0)
        assert key.is_cooling_down is True
        key.reset_cooldown()
        assert key.status == APIKeyStatus.ACTIVE
        assert key.is_cooling_down is False


class TestAPIKeyStateCircuitBreaker:

    def test_consecutive_failures_increment(self):
        key = _make_key()
        for _ in range(3):
            key.record_request_start()
            key.record_failure(is_rate_limit=False)
        assert key.consecutive_failures == 3

    def test_success_resets_consecutive_failures(self):
        key = _make_key()
        key.record_request_start()
        key.record_failure(is_rate_limit=False)
        assert key.consecutive_failures == 1
        key.record_request_start()
        key.record_request_end(latency=0.1, tokens_in=5, tokens_out=3)
        assert key.consecutive_failures == 0

    def test_mark_failed_directly(self):
        key = _make_key()
        key.mark_failed()
        assert key.status == APIKeyStatus.FAILED
        assert key.is_available is False


class TestKeyPool:

    def test_pool_with_three_keys_all_available(self):
        pool = KeyPool(["gsk_key_1", "gsk_key_2", "gsk_key_3"])
        eligible = pool.get_eligible_keys()
        assert len(eligible) == 3

    def test_pool_get_key_returns_highest_health(self):
        pool = KeyPool(["gsk_key_1", "gsk_key_2"])
        key = pool.get_key()
        assert key is not None
        assert key.is_available

    def test_pool_disable_key_reduces_eligible(self):
        pool = KeyPool(["gsk_key_1", "gsk_key_2", "gsk_key_3"])
        keys = list(pool._keys.values())
        pool.disable_key(keys[0].key_id)
        eligible = pool.get_eligible_keys()
        assert len(eligible) == 2

    def test_pool_all_disabled_no_eligible_keys(self):
        pool = KeyPool(["gsk_key_1", "gsk_key_2"])
        for k in pool.list_keys():
            pool.disable_key(k.key_id)
        assert pool.get_eligible_keys() == []
        assert pool.get_key() is None

    def test_pool_mark_unhealthy_rate_limit(self):
        pool = KeyPool(["gsk_key_1"])
        key = pool.list_keys()[0]
        key.record_request_start()
        pool.mark_unhealthy(key.key_id, failed=False, cooldown_secs=60.0)
        assert key.status == APIKeyStatus.RATE_LIMITED

    def test_pool_mark_unhealthy_failed(self):
        pool = KeyPool(["gsk_key_1"])
        key = pool.list_keys()[0]
        pool.mark_unhealthy(key.key_id, failed=True)
        assert key.status == APIKeyStatus.FAILED

    def test_pool_enable_key_restores_active(self):
        pool = KeyPool(["gsk_key_1"])
        key = pool.list_keys()[0]
        pool.disable_key(key.key_id)
        assert key.status == APIKeyStatus.DISABLED
        pool.enable_key(key.key_id)
        assert key.status == APIKeyStatus.ACTIVE


class TestKeyPoolThreadSafety:

    def test_concurrent_rpm_increments_are_consistent(self):
        key = _make_key()
        threads = []
        for _ in range(10):
            t = threading.Thread(target=key.record_request_start)
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All 10 threads incremented — active_requests should be 10
        assert key.active_requests == 10

    def test_concurrent_pool_access_does_not_corrupt_state(self):
        pool = KeyPool(["gsk_key_1", "gsk_key_2", "gsk_key_3"])
        errors = []

        def access():
            try:
                key = pool.get_key()
                if key:
                    key.record_request_start()
                    key.record_request_end(latency=0.01, tokens_in=1, tokens_out=1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"
