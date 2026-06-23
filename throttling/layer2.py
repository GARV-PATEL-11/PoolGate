"""Layer-2 model-level throttle: token-bucket RPM + concurrency."""

from __future__ import annotations

from throttling.algorithms import ConcurrencyCounter, TokenBucket
from throttling.config import ThrottleConfig
from throttling.types import ThrottleMode


class ModelThrottle:
    """Per-model token-bucket RPM + concurrency throttle (layer 2)."""

    def __init__(self, config: ThrottleConfig) -> None:
        self._config = config
        self._rpm_buckets: dict[str, TokenBucket] = {}
        self._concurrent_counters: dict[str, ConcurrencyCounter] = {}
        for model_id, model_cfg in config.model_configs.items():
            self._rpm_buckets[model_id] = TokenBucket(model_cfg.max_rpm)
            self._concurrent_counters[model_id] = ConcurrencyCounter()

    def has_config(self, model: str) -> bool:
        return model in self._config.model_configs

    def try_acquire_rpm(self, model: str) -> bool:
        cfg = self._config.model_configs.get(model)
        if cfg is None or cfg.mode in (ThrottleMode.DISABLED, ThrottleMode.CONCURRENCY_ONLY):
            return True
        bucket = self._rpm_buckets.get(model)
        return bucket.try_acquire() if bucket is not None else True

    def try_acquire_concurrent(self, model: str) -> bool:
        cfg = self._config.model_configs.get(model)
        if cfg is None or cfg.mode in (ThrottleMode.DISABLED, ThrottleMode.RPM_ONLY):
            return True
        counter = self._concurrent_counters.get(model)
        return counter.try_acquire(cfg.max_concurrent) if counter is not None else True

    def release_concurrent(self, model: str) -> None:
        counter = self._concurrent_counters.get(model)
        if counter is not None:
            counter.release()

    def concurrent_count(self, model: str) -> int:
        counter = self._concurrent_counters.get(model)
        return counter.count() if counter is not None else 0
