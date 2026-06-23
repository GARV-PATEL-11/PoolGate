"""Configuration dataclasses for the two-layer throttling system."""

from __future__ import annotations

from dataclasses import dataclass, field

from throttling.types import CapabilityType, ThrottleMode


@dataclass
class CapabilityConfig:
    """Layer-1 throttle configuration for one capability type."""

    max_rpm: int
    max_concurrent: int
    burst_multiplier: float = 1.0
    mode: ThrottleMode = ThrottleMode.RPM_AND_CONCURRENCY

    @property
    def effective_rpm(self) -> int:
        return int(self.max_rpm * self.burst_multiplier)


@dataclass
class ModelConfig:
    """Layer-2 throttle configuration for one model."""

    max_rpm: int
    max_concurrent: int
    mode: ThrottleMode = ThrottleMode.RPM_AND_CONCURRENCY


@dataclass
class ThrottleConfig:
    """Top-level throttle config. Passed to ThrottleMiddleware."""

    enabled: bool = True
    capability_configs: dict[CapabilityType, CapabilityConfig] = field(default_factory=dict)
    model_configs: dict[str, ModelConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.capability_configs:
            self.capability_configs = _DEFAULT_CAPABILITY_CONFIGS.copy()


_DEFAULT_CAPABILITY_CONFIGS: dict[CapabilityType, CapabilityConfig] = {
    CapabilityType.TEXT_GENERATION: CapabilityConfig(max_rpm=25, max_concurrent=10, burst_multiplier=1.2),
    CapabilityType.STRUCTURED_GEN: CapabilityConfig(max_rpm=20, max_concurrent=8),
    CapabilityType.TOOL_CALLING: CapabilityConfig(max_rpm=20, max_concurrent=8),
    CapabilityType.MODERATION: CapabilityConfig(max_rpm=50, max_concurrent=20),
    CapabilityType.TRANSCRIPTION: CapabilityConfig(max_rpm=10, max_concurrent=4, mode=ThrottleMode.CONCURRENCY_ONLY),
    CapabilityType.SYNTHESIS: CapabilityConfig(max_rpm=5, max_concurrent=3, mode=ThrottleMode.CONCURRENCY_ONLY),
}
