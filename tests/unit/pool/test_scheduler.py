"""Unit tests for poolgate/pool/scheduler.py."""

from __future__ import annotations

import pytest

from poolgate.core.logger import LoggerManager
from poolgate.core.paths import PathConfig
from poolgate.exceptions.keys import NoAvailableAPIKeyError
from poolgate.pool.scheduler import RequestScheduler
from poolgate.schemas.common.runtime import APIKeyStatus


@pytest.fixture
def logger():
    return LoggerManager(PathConfig(base_dir=None), "INFO").get()


@pytest.fixture
def scheduler(three_keys, groq_config, logger):
    return RequestScheduler(three_keys, groq_config, logger)


class TestAcquireRelease:

    def test_acquire_key_returns_an_available_key(self, scheduler):
        key = scheduler.acquire_key("req-1")
        assert key.active_requests == 1

    def test_acquire_key_raises_when_pool_exhausted(self, three_keys, groq_config, logger):
        for k in three_keys:
            k.mark_disabled()
        scheduler = RequestScheduler(three_keys, groq_config, logger)
        with pytest.raises(NoAvailableAPIKeyError):
            scheduler.acquire_key("req-1")

    def test_release_key_records_usage(self, scheduler):
        key = scheduler.acquire_key("req-1")
        scheduler.release_key(key, latency=0.2, tokens_in=10, tokens_out=5)
        assert key.active_requests == 0
        assert key.input_tokens == 10
        assert key.output_tokens == 5


class TestModelAwareRpm:

    def test_acquire_key_with_unknown_model_falls_back_silently(self, scheduler):
        key = scheduler.acquire_key("req-1", model="totally-not-a-real-model")
        assert key is not None

    def test_acquire_key_with_known_model_does_not_raise(self, scheduler):
        key = scheduler.acquire_key("req-1", model="whisper-large-v3")
        assert key is not None

    def test_max_rpm_for_model_returns_real_limit_for_known_model(self, scheduler):
        rpm = scheduler._max_rpm_for_model("llama-3.3-70b-versatile")
        assert isinstance(rpm, int)
        assert rpm > 0

    def test_max_rpm_for_model_returns_none_for_unknown_model(self, scheduler):
        assert scheduler._max_rpm_for_model("not-a-real-model") is None


class TestCircuitBreakerTrip:

    def test_key_trips_to_failed_after_threshold_consecutive_failures(
        self,
        three_keys,
        groq_config,
        logger,
    ):
        groq_config.failure_threshold = 3
        scheduler = RequestScheduler(three_keys, groq_config, logger)
        target = three_keys[0]

        for _ in range(3):
            scheduler.mark_key_failure(target, is_rate_limit=False)

        assert target.status == APIKeyStatus.FAILED
        assert target.is_available is False

    def test_rate_limit_failures_do_not_trip_the_breaker(self, three_keys, groq_config, logger):
        """
        Rate limits already have their own cooldown-based recovery — a key
        that's merely rate-limited is busy, not unhealthy, so consecutive
        429s alone must never trip it to FAILED.
        """
        groq_config.failure_threshold = 2
        scheduler = RequestScheduler(three_keys, groq_config, logger)
        target = three_keys[0]

        for _ in range(5):
            scheduler.mark_key_failure(target, is_rate_limit=True)

        assert target.status != APIKeyStatus.FAILED

    def test_success_resets_the_breaker_counter(self, three_keys, groq_config, logger):
        groq_config.failure_threshold = 3
        scheduler = RequestScheduler(three_keys, groq_config, logger)
        target = three_keys[0]

        scheduler.mark_key_failure(target, is_rate_limit=False)
        scheduler.mark_key_failure(target, is_rate_limit=False)
        target.record_request_start()
        target.record_request_end(latency=0.1, tokens_in=1, tokens_out=1)

        scheduler.mark_key_failure(target, is_rate_limit=False)
        assert target.status != APIKeyStatus.FAILED  # counter reset by the success above
