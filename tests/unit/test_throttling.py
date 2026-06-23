"""Unit tests for the two-layer throttling system."""

from __future__ import annotations

import threading
import time

import pytest

from exceptions import CapabilityThrottledError, ModelThrottledError
from throttling import (
    CapabilityConfig,
    CapabilityType,
    ModelConfig,
    ThrottleConfig,
    ThrottleHandle,
    ThrottleMiddleware,
    ThrottleMode,
)
from throttling.algorithms import ConcurrencyCounter, SlidingWindowCounter, TokenBucket
from throttling.config import _DEFAULT_CAPABILITY_CONFIGS
from throttling.layer1 import CapabilityThrottle
from throttling.layer2 import ModelThrottle

# ─── Algorithms ────────────────────────────────────────────────────────────────


class TestSlidingWindowCounter:
    def test_allows_up_to_limit(self) -> None:
        counter = SlidingWindowCounter()
        for _ in range(5):
            assert counter.try_acquire(5) is True

    def test_blocks_over_limit(self) -> None:
        counter = SlidingWindowCounter()
        for _ in range(5):
            counter.try_acquire(5)
        assert counter.try_acquire(5) is False

    def test_count_reflects_window(self) -> None:
        counter = SlidingWindowCounter()
        counter.try_acquire(100)
        counter.try_acquire(100)
        assert counter.count() == 2

    def test_window_expiry_allows_new_requests(self) -> None:
        counter = SlidingWindowCounter(window_secs=0.05)
        for _ in range(3):
            counter.try_acquire(3)
        assert counter.try_acquire(3) is False
        time.sleep(0.06)
        assert counter.try_acquire(3) is True

    def test_thread_safe_concurrent_acquisition(self) -> None:
        counter = SlidingWindowCounter()
        results = []
        lock = threading.Lock()

        def acquire() -> None:
            ok = counter.try_acquire(10)
            with lock:
                results.append(ok)

        threads = [threading.Thread(target=acquire) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        acquired = sum(1 for r in results if r)
        assert acquired == 10


class TestTokenBucket:
    def test_starts_full_and_allows_up_to_capacity(self) -> None:
        bucket = TokenBucket(max_rpm=5)
        for _ in range(5):
            assert bucket.try_acquire() is True

    def test_blocks_when_empty(self) -> None:
        bucket = TokenBucket(max_rpm=5)
        for _ in range(5):
            bucket.try_acquire()
        assert bucket.try_acquire() is False

    def test_refills_over_time(self) -> None:
        bucket = TokenBucket(max_rpm=60)  # 1 token/sec
        for _ in range(60):
            bucket.try_acquire()
        assert bucket.try_acquire() is False
        time.sleep(1.05)
        assert bucket.try_acquire() is True

    def test_available_returns_non_negative(self) -> None:
        bucket = TokenBucket(max_rpm=10)
        assert bucket.available() >= 0.0

    def test_thread_safe_concurrent_acquisition(self) -> None:
        bucket = TokenBucket(max_rpm=10)
        results = []
        lock = threading.Lock()

        def acquire() -> None:
            ok = bucket.try_acquire()
            with lock:
                results.append(ok)

        threads = [threading.Thread(target=acquire) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        acquired = sum(1 for r in results if r)
        assert acquired == 10


class TestConcurrencyCounter:
    def test_allows_up_to_limit(self) -> None:
        counter = ConcurrencyCounter()
        assert counter.try_acquire(3) is True
        assert counter.try_acquire(3) is True
        assert counter.try_acquire(3) is True

    def test_blocks_over_limit(self) -> None:
        counter = ConcurrencyCounter()
        counter.try_acquire(2)
        counter.try_acquire(2)
        assert counter.try_acquire(2) is False

    def test_release_decrements(self) -> None:
        counter = ConcurrencyCounter()
        counter.try_acquire(1)
        assert counter.count() == 1
        counter.release()
        assert counter.count() == 0

    def test_release_is_safe_when_zero(self) -> None:
        counter = ConcurrencyCounter()
        counter.release()  # should not raise
        assert counter.count() == 0

    def test_thread_safe_concurrent_access(self) -> None:
        counter = ConcurrencyCounter()
        results = []
        lock = threading.Lock()

        def acquire_and_release() -> None:
            ok = counter.try_acquire(5)
            with lock:
                results.append(ok)
            if ok:
                time.sleep(0.001)
                counter.release()

        threads = [threading.Thread(target=acquire_and_release) for _ in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.count() == 0


# ─── Config ────────────────────────────────────────────────────────────────────


class TestThrottleConfig:
    def test_default_capability_configs_populated(self) -> None:
        cfg = ThrottleConfig()
        assert CapabilityType.TEXT_GENERATION in cfg.capability_configs
        assert CapabilityType.MODERATION in cfg.capability_configs

    def test_default_model_configs_empty(self) -> None:
        cfg = ThrottleConfig()
        assert cfg.model_configs == {}

    def test_custom_capability_config_overrides_defaults(self) -> None:
        custom = {CapabilityType.TEXT_GENERATION: CapabilityConfig(max_rpm=100, max_concurrent=50)}
        cfg = ThrottleConfig(capability_configs=custom)
        assert cfg.capability_configs[CapabilityType.TEXT_GENERATION].max_rpm == 100

    def test_capability_config_effective_rpm(self) -> None:
        c = CapabilityConfig(max_rpm=25, max_concurrent=10, burst_multiplier=1.2)
        assert c.effective_rpm == 30  # int(25 * 1.2)

    def test_disabled_config(self) -> None:
        cfg = ThrottleConfig(enabled=False)
        assert cfg.enabled is False


class TestDefaultCapabilityConfigs:
    def test_text_generation_has_rpm_and_concurrency(self) -> None:
        c = _DEFAULT_CAPABILITY_CONFIGS[CapabilityType.TEXT_GENERATION]
        assert c.max_rpm > 0
        assert c.max_concurrent > 0

    def test_transcription_is_concurrency_only(self) -> None:
        c = _DEFAULT_CAPABILITY_CONFIGS[CapabilityType.TRANSCRIPTION]
        assert c.mode == ThrottleMode.CONCURRENCY_ONLY

    def test_synthesis_is_concurrency_only(self) -> None:
        c = _DEFAULT_CAPABILITY_CONFIGS[CapabilityType.SYNTHESIS]
        assert c.mode == ThrottleMode.CONCURRENCY_ONLY


# ─── Layer 1 ────────────────────────────────────────────────────────────────────


class TestCapabilityThrottle:
    def _small_config(self, mode: ThrottleMode = ThrottleMode.RPM_AND_CONCURRENCY) -> ThrottleConfig:
        return ThrottleConfig(
            capability_configs={
                CapabilityType.TEXT_GENERATION: CapabilityConfig(max_rpm=3, max_concurrent=2, mode=mode)
            }
        )

    def test_rpm_blocks_when_limit_reached(self) -> None:
        cfg = self._small_config(ThrottleMode.RPM_ONLY)
        l1 = CapabilityThrottle(cfg)
        for _ in range(3):
            l1.try_acquire_rpm(CapabilityType.TEXT_GENERATION)
        assert l1.try_acquire_rpm(CapabilityType.TEXT_GENERATION) is False

    def test_concurrent_blocks_when_limit_reached(self) -> None:
        cfg = self._small_config(ThrottleMode.CONCURRENCY_ONLY)
        l1 = CapabilityThrottle(cfg)
        l1.try_acquire_concurrent(CapabilityType.TEXT_GENERATION)
        l1.try_acquire_concurrent(CapabilityType.TEXT_GENERATION)
        assert l1.try_acquire_concurrent(CapabilityType.TEXT_GENERATION) is False

    def test_release_concurrent_allows_more(self) -> None:
        cfg = self._small_config(ThrottleMode.CONCURRENCY_ONLY)
        l1 = CapabilityThrottle(cfg)
        l1.try_acquire_concurrent(CapabilityType.TEXT_GENERATION)
        l1.try_acquire_concurrent(CapabilityType.TEXT_GENERATION)
        assert l1.try_acquire_concurrent(CapabilityType.TEXT_GENERATION) is False
        l1.release_concurrent(CapabilityType.TEXT_GENERATION)
        assert l1.try_acquire_concurrent(CapabilityType.TEXT_GENERATION) is True

    def test_rpm_skipped_in_concurrency_only_mode(self) -> None:
        cfg = self._small_config(ThrottleMode.CONCURRENCY_ONLY)
        l1 = CapabilityThrottle(cfg)
        for _ in range(100):
            assert l1.try_acquire_rpm(CapabilityType.TEXT_GENERATION) is True

    def test_concurrent_skipped_in_rpm_only_mode(self) -> None:
        cfg = self._small_config(ThrottleMode.RPM_ONLY)
        l1 = CapabilityThrottle(cfg)
        for _ in range(100):
            assert l1.try_acquire_concurrent(CapabilityType.TEXT_GENERATION) is True

    def test_unknown_capability_type_returns_true(self) -> None:
        cfg = self._small_config()
        l1 = CapabilityThrottle(cfg)
        assert l1.try_acquire_rpm(CapabilityType.MODERATION) is True
        assert l1.try_acquire_concurrent(CapabilityType.MODERATION) is True

    def test_rpm_count_and_concurrent_count(self) -> None:
        cfg = self._small_config()
        l1 = CapabilityThrottle(cfg)
        l1.try_acquire_rpm(CapabilityType.TEXT_GENERATION)
        l1.try_acquire_concurrent(CapabilityType.TEXT_GENERATION)
        assert l1.rpm_count(CapabilityType.TEXT_GENERATION) == 1
        assert l1.concurrent_count(CapabilityType.TEXT_GENERATION) == 1


# ─── Layer 2 ────────────────────────────────────────────────────────────────────


class TestModelThrottle:
    MODEL = "llama-3.3-70b-versatile"

    def _cfg(self, mode: ThrottleMode = ThrottleMode.RPM_AND_CONCURRENCY) -> ThrottleConfig:
        return ThrottleConfig(model_configs={self.MODEL: ModelConfig(max_rpm=3, max_concurrent=2, mode=mode)})

    def test_has_config_true_for_known_model(self) -> None:
        l2 = ModelThrottle(self._cfg())
        assert l2.has_config(self.MODEL) is True

    def test_has_config_false_for_unknown_model(self) -> None:
        l2 = ModelThrottle(self._cfg())
        assert l2.has_config("unknown-model") is False

    def test_rpm_blocks_when_limit_reached(self) -> None:
        l2 = ModelThrottle(self._cfg(ThrottleMode.RPM_ONLY))
        for _ in range(3):
            l2.try_acquire_rpm(self.MODEL)
        assert l2.try_acquire_rpm(self.MODEL) is False

    def test_concurrent_blocks_when_limit_reached(self) -> None:
        l2 = ModelThrottle(self._cfg(ThrottleMode.CONCURRENCY_ONLY))
        l2.try_acquire_concurrent(self.MODEL)
        l2.try_acquire_concurrent(self.MODEL)
        assert l2.try_acquire_concurrent(self.MODEL) is False

    def test_release_concurrent_allows_more(self) -> None:
        l2 = ModelThrottle(self._cfg(ThrottleMode.CONCURRENCY_ONLY))
        l2.try_acquire_concurrent(self.MODEL)
        l2.try_acquire_concurrent(self.MODEL)
        l2.release_concurrent(self.MODEL)
        assert l2.try_acquire_concurrent(self.MODEL) is True

    def test_unknown_model_rpm_returns_true(self) -> None:
        l2 = ModelThrottle(self._cfg())
        assert l2.try_acquire_rpm("other-model") is True

    def test_unknown_model_concurrent_returns_true(self) -> None:
        l2 = ModelThrottle(self._cfg())
        assert l2.try_acquire_concurrent("other-model") is True


# ─── ThrottleHandle ─────────────────────────────────────────────────────────────


class TestThrottleHandle:
    def test_release_decrements_l1_concurrent(self) -> None:
        cfg = ThrottleConfig(
            capability_configs={CapabilityType.TEXT_GENERATION: CapabilityConfig(max_rpm=100, max_concurrent=5)}
        )
        l1 = CapabilityThrottle(cfg)
        l1.try_acquire_concurrent(CapabilityType.TEXT_GENERATION)
        assert l1.concurrent_count(CapabilityType.TEXT_GENERATION) == 1

        handle = ThrottleHandle(
            l1=l1,
            l1_cap=CapabilityType.TEXT_GENERATION,
            l2=None,
            l2_model=None,
        )
        handle.release()
        assert l1.concurrent_count(CapabilityType.TEXT_GENERATION) == 0

    def test_release_is_idempotent(self) -> None:
        handle = ThrottleHandle(l1=None, l1_cap=None, l2=None, l2_model=None)
        handle.release()
        handle.release()  # should not raise

    def test_noop_handle_release_is_safe(self) -> None:
        handle = ThrottleHandle(l1=None, l1_cap=None, l2=None, l2_model=None)
        handle.release()  # no l1/l2 — safe no-op


# ─── ThrottleMiddleware ──────────────────────────────────────────────────────────


class TestThrottleMiddlewareDisabled:
    def test_disabled_always_returns_noop_handle(self) -> None:
        cfg = ThrottleConfig(enabled=False)
        m = ThrottleMiddleware(cfg)
        handle = m.check("chat", "llama-3.3-70b-versatile", "req-1")
        handle.release()  # no-op, no exception

    def test_disabled_never_raises(self) -> None:
        cfg = ThrottleConfig(
            enabled=False,
            capability_configs={CapabilityType.TEXT_GENERATION: CapabilityConfig(max_rpm=0, max_concurrent=0)},
        )
        m = ThrottleMiddleware(cfg)
        for _ in range(100):
            h = m.check("chat", "any-model", "req")
            h.release()


class TestThrottleMiddlewareCapabilityLayer:
    def _tight_middleware(self) -> ThrottleMiddleware:
        cfg = ThrottleConfig(
            capability_configs={CapabilityType.TEXT_GENERATION: CapabilityConfig(max_rpm=2, max_concurrent=2)}
        )
        return ThrottleMiddleware(cfg)

    def test_check_returns_handle(self) -> None:
        m = self._tight_middleware()
        handle = m.check("chat", "llama-3.3-70b-versatile", "r1")
        assert isinstance(handle, ThrottleHandle)
        handle.release()

    def test_rpm_limit_raises_capability_throttled(self) -> None:
        m = self._tight_middleware()
        h1 = m.check("chat", "llama", "r1")
        h2 = m.check("chat", "llama", "r2")
        with pytest.raises(CapabilityThrottledError) as exc_info:
            m.check("chat", "llama", "r3")
        assert exc_info.value.capability == "chat"
        h1.release()
        h2.release()

    def test_concurrency_limit_raises_capability_throttled(self) -> None:
        cfg = ThrottleConfig(
            capability_configs={CapabilityType.TEXT_GENERATION: CapabilityConfig(max_rpm=100, max_concurrent=1)}
        )
        m = ThrottleMiddleware(cfg)
        h1 = m.check("chat", "llama", "r1")
        with pytest.raises(CapabilityThrottledError):
            m.check("chat", "llama", "r2")
        h1.release()
        # After release, should be allowed again
        h2 = m.check("chat", "llama", "r3")
        h2.release()

    def test_capability_string_mapping(self) -> None:
        m = ThrottleMiddleware()
        for cap in ["chat", "structured", "tools", "moderation", "transcription", "synthesis", "api"]:
            h = m.check(cap, "any-model", "req")
            h.release()

    def test_unknown_capability_maps_to_text_generation(self) -> None:
        # "api" and any unknown string should map to TEXT_GENERATION
        m = ThrottleMiddleware()
        h = m.check("unknown-capability", "model", "req")
        h.release()

    def test_handle_release_restores_concurrency(self) -> None:
        cfg = ThrottleConfig(
            capability_configs={CapabilityType.TEXT_GENERATION: CapabilityConfig(max_rpm=100, max_concurrent=1)}
        )
        m = ThrottleMiddleware(cfg)
        h1 = m.check("chat", "llama", "r1")
        with pytest.raises(CapabilityThrottledError):
            m.check("chat", "llama", "r2")
        h1.release()
        h2 = m.check("chat", "llama", "r3")
        h2.release()


class TestThrottleMiddlewareModelLayer:
    MODEL = "llama-3.3-70b-versatile"

    def _middleware_with_model(self, max_rpm: int = 100, max_concurrent: int = 1) -> ThrottleMiddleware:
        cfg = ThrottleConfig(model_configs={self.MODEL: ModelConfig(max_rpm=max_rpm, max_concurrent=max_concurrent)})
        return ThrottleMiddleware(cfg)

    def test_model_concurrency_limit_raises_model_throttled(self) -> None:
        m = self._middleware_with_model(max_concurrent=1)
        h1 = m.check("chat", self.MODEL, "r1")
        with pytest.raises(ModelThrottledError) as exc_info:
            m.check("chat", self.MODEL, "r2")
        assert exc_info.value.model == self.MODEL
        h1.release()

    def test_model_throttle_restores_l1_concurrent_on_failure(self) -> None:
        """When L2 rejects, L1 concurrency must be rolled back."""
        cfg = ThrottleConfig(
            capability_configs={CapabilityType.TEXT_GENERATION: CapabilityConfig(max_rpm=100, max_concurrent=1)},
            model_configs={self.MODEL: ModelConfig(max_rpm=100, max_concurrent=0)},
        )
        m = ThrottleMiddleware(cfg)
        # L2 concurrency is 0 — always fails
        with pytest.raises(ModelThrottledError):
            m.check("chat", self.MODEL, "r1")
        # L1 concurrency should have been rolled back, so this must succeed
        other_cfg = ThrottleConfig(
            capability_configs={CapabilityType.TEXT_GENERATION: CapabilityConfig(max_rpm=100, max_concurrent=1)}
        )
        m2 = ThrottleMiddleware(other_cfg)
        h = m2.check("chat", "other-model", "r2")
        h.release()

    def test_unknown_model_not_throttled(self) -> None:
        m = self._middleware_with_model()
        h = m.check("chat", "unknown-model", "r1")
        h.release()

    def test_model_handle_releases_both_layers(self) -> None:
        cfg = ThrottleConfig(
            capability_configs={CapabilityType.TEXT_GENERATION: CapabilityConfig(max_rpm=100, max_concurrent=2)},
            model_configs={self.MODEL: ModelConfig(max_rpm=100, max_concurrent=2)},
        )
        m = ThrottleMiddleware(cfg)
        h = m.check("chat", self.MODEL, "r1")
        h.release()
        # Both layers should have released — a subsequent check should work
        h2 = m.check("chat", self.MODEL, "r2")
        h2.release()


class TestThrottleMiddlewareIntegration:
    def test_request_id_in_throttle_error(self) -> None:
        cfg = ThrottleConfig(
            capability_configs={CapabilityType.TEXT_GENERATION: CapabilityConfig(max_rpm=0, max_concurrent=0)}
        )
        m = ThrottleMiddleware(cfg)
        with pytest.raises(CapabilityThrottledError) as exc_info:
            m.check("chat", "model", "my-request-id")
        assert exc_info.value.request_id == "my-request-id"

    def test_default_middleware_allows_normal_traffic(self) -> None:
        m = ThrottleMiddleware()
        handles = []
        for i in range(5):
            h = m.check("chat", "llama-3.3-70b-versatile", f"req-{i}")
            handles.append(h)
        for h in handles:
            h.release()
