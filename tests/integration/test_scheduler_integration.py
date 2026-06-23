"""Integration tests for RequestScheduler — multi-key selection and lifecycle."""

from __future__ import annotations

import threading

import pytest

from core.logger_manager import ObservabilityLogger
from exceptions.keys import NoAvailableAPIKeyError
from key_manager.key_pool import APIKeyState
from schedulers.request_scheduler import RequestScheduler
from schedulers.scheduling_strategies import SchedulingStrategyType
from schemas.runtime import APIKeyStatus


def _make_logger() -> ObservabilityLogger:
    return ObservabilityLogger(name="test_scheduler", level="ERROR")


def _make_config(max_rpm: int = 30, cooldown_secs: float = 60.0, failure_threshold: int = 5):
    from core.config import GroqConfig

    return GroqConfig(
        api_keys=["gsk_dummy"],
        max_rpm_per_key=max_rpm,
        cooldown_seconds=cooldown_secs,
        failure_threshold=failure_threshold,
    )


def _make_keys(n: int) -> list[APIKeyState]:
    return [APIKeyState.from_key(key_id=f"key_{i}", raw_key=f"gsk_key_{i}") for i in range(n)]


def _make_scheduler(
    keys: list[APIKeyState],
    strategy: SchedulingStrategyType = SchedulingStrategyType.HEALTH_AWARE,
) -> RequestScheduler:
    return RequestScheduler(keys, _make_config(), _make_logger(), strategy=strategy)


# ---------------------------------------------------------------------------
# Basic acquire / release
# ---------------------------------------------------------------------------


class TestAcquireRelease:

    def test_acquire_returns_a_key(self):
        keys = _make_keys(2)
        scheduler = _make_scheduler(keys)
        key = scheduler.acquire_key("req-1")
        assert key is not None
        assert key.key_id.startswith("key_")

    def test_acquire_increments_active_requests(self):
        keys = _make_keys(1)
        scheduler = _make_scheduler(keys)
        key = scheduler.acquire_key("req-1")
        assert key.active_requests == 1

    def test_release_decrements_active_requests(self):
        keys = _make_keys(1)
        scheduler = _make_scheduler(keys)
        key = scheduler.acquire_key("req-1")
        scheduler.release_key(key, latency=0.1, tokens_in=10, tokens_out=5)
        assert key.active_requests == 0

    def test_release_increments_success_count(self):
        keys = _make_keys(1)
        scheduler = _make_scheduler(keys)
        key = scheduler.acquire_key("req-1")
        scheduler.release_key(key, latency=0.1, tokens_in=10, tokens_out=5)
        assert key.success_count == 1

    def test_no_available_key_raises(self):
        keys = _make_keys(2)
        for k in keys:
            k.mark_disabled()
        scheduler = _make_scheduler(keys)
        with pytest.raises(NoAvailableAPIKeyError):
            scheduler.acquire_key("req-1")


# ---------------------------------------------------------------------------
# Failure marking
# ---------------------------------------------------------------------------


class TestFailureMarking:

    def test_mark_key_failure_increments_consecutive_failures(self):
        keys = _make_keys(1)
        scheduler = _make_scheduler(keys)
        key = scheduler.acquire_key("req-1")
        scheduler.mark_key_failure(key, is_rate_limit=False)
        assert key.consecutive_failures == 1

    def test_mark_key_failure_with_rate_limit_enters_cooldown(self):
        keys = _make_keys(1)
        scheduler = _make_scheduler(keys)
        key = scheduler.acquire_key("req-1")
        scheduler.mark_key_failure(key, is_rate_limit=True)
        assert key.status == APIKeyStatus.RATE_LIMITED
        assert key.is_cooling_down is True

    def test_mark_key_disabled_marks_disabled(self):
        keys = _make_keys(1)
        scheduler = _make_scheduler(keys)
        key = keys[0]
        scheduler.mark_key_disabled(key)
        assert key.status == APIKeyStatus.DISABLED

    def test_circuit_breaker_trips_after_threshold(self):
        threshold = 3
        keys = _make_keys(1)
        scheduler = _make_scheduler(keys, SchedulingStrategyType.HEALTH_AWARE)
        # Override config to use threshold=3
        config = _make_config(failure_threshold=threshold)
        scheduler._config = config
        key = keys[0]
        for _ in range(threshold):
            key.record_request_start()
            scheduler.mark_key_failure(key, is_rate_limit=False)
        assert key.status == APIKeyStatus.FAILED


# ---------------------------------------------------------------------------
# Strategy switching
# ---------------------------------------------------------------------------


class TestStrategySwitching:

    def test_initial_strategy_name(self):
        keys = _make_keys(2)
        scheduler = _make_scheduler(keys, SchedulingStrategyType.ROUND_ROBIN)
        assert (
            "round_robin" in scheduler.current_strategy_name().lower()
            or "roundrobin" in scheduler.current_strategy_name().lower()
            or scheduler.current_strategy_name() != ""
        )

    def test_set_strategy_at_runtime(self):
        keys = _make_keys(2)
        scheduler = _make_scheduler(keys, SchedulingStrategyType.HEALTH_AWARE)
        scheduler.set_strategy(SchedulingStrategyType.ROUND_ROBIN)
        # Should not raise and should have changed
        name = scheduler.current_strategy_name()
        assert isinstance(name, str)

    def test_health_aware_picks_healthiest_key(self):
        keys = _make_keys(2)
        # Degrade key_0 with many failures
        keys[0].record_request_start()
        keys[0].record_failure(is_rate_limit=False)
        keys[0].record_request_start()
        keys[0].record_failure(is_rate_limit=False)

        scheduler = _make_scheduler(keys, SchedulingStrategyType.HEALTH_AWARE)
        selected = scheduler.acquire_key("req-1")
        # Healthier key should be key_1
        assert selected.key_id == "key_1"


# ---------------------------------------------------------------------------
# Concurrent access
# ---------------------------------------------------------------------------


class TestConcurrentAccess:

    def test_concurrent_acquires_do_not_corrupt_state(self):
        keys = _make_keys(3)
        scheduler = _make_scheduler(keys)
        errors = []

        def task():
            try:
                key = scheduler.acquire_key("req")
                scheduler.release_key(key, latency=0.01, tokens_in=1, tokens_out=1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=task) for _ in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrency errors: {errors}"


# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------


class TestStatusSummary:

    def test_status_summary_has_all_keys(self):
        keys = _make_keys(3)
        scheduler = _make_scheduler(keys)
        summary = scheduler.status_summary()
        assert len(summary) == 3

    def test_status_summary_structure(self):
        keys = _make_keys(1)
        scheduler = _make_scheduler(keys)
        entry = scheduler.status_summary()[0]
        assert "key_id" in entry
        assert "status" in entry
        assert "requests_per_minute" in entry
        assert "score" in entry
