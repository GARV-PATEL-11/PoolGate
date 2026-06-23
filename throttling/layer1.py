"""Layer-1 capability-level throttle: sliding window RPM + concurrency."""

from __future__ import annotations

from throttling.algorithms import ConcurrencyCounter, SlidingWindowCounter
from throttling.config import ThrottleConfig
from throttling.types import CapabilityType, ThrottleMode


class CapabilityThrottle:
    """Per-capability sliding-window RPM + concurrency throttle (layer 1)."""

    def __init__(self, config: ThrottleConfig) -> None:
        self._config = config
        self._rpm_counters: dict[CapabilityType, SlidingWindowCounter] = {}
        self._concurrent_counters: dict[CapabilityType, ConcurrencyCounter] = {}
        for cap_type in config.capability_configs:
            self._rpm_counters[cap_type] = SlidingWindowCounter()
            self._concurrent_counters[cap_type] = ConcurrencyCounter()

    def try_acquire_rpm(self, cap_type: CapabilityType) -> bool:
        cfg = self._config.capability_configs.get(cap_type)
        if cfg is None or cfg.mode in (ThrottleMode.DISABLED, ThrottleMode.CONCURRENCY_ONLY):
            return True
        counter = self._rpm_counters.get(cap_type)
        return counter.try_acquire(cfg.effective_rpm) if counter is not None else True

    def try_acquire_concurrent(self, cap_type: CapabilityType) -> bool:
        cfg = self._config.capability_configs.get(cap_type)
        if cfg is None or cfg.mode in (ThrottleMode.DISABLED, ThrottleMode.RPM_ONLY):
            return True
        counter = self._concurrent_counters.get(cap_type)
        return counter.try_acquire(cfg.max_concurrent) if counter is not None else True

    def release_concurrent(self, cap_type: CapabilityType) -> None:
        counter = self._concurrent_counters.get(cap_type)
        if counter is not None:
            counter.release()

    def rpm_count(self, cap_type: CapabilityType) -> int:
        counter = self._rpm_counters.get(cap_type)
        return counter.count() if counter is not None else 0

    def concurrent_count(self, cap_type: CapabilityType) -> int:
        counter = self._concurrent_counters.get(cap_type)
        return counter.count() if counter is not None else 0
