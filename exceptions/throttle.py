"""Throttle-related exceptions for the two-layer throttling system."""

from __future__ import annotations

from exceptions.base import GroqServiceError


class ThrottleError(GroqServiceError):
    """Base for all throttle errors."""


class CapabilityThrottledError(ThrottleError):
    """Raised when the capability-level (layer-1) limit is exceeded."""

    def __init__(self, capability: str, *, request_id: str | None = None) -> None:
        self.capability = capability
        super().__init__(
            f"Capability '{capability}' is currently throttled — too many concurrent or per-minute requests.",
            request_id=request_id,
        )


class ModelThrottledError(ThrottleError):
    """Raised when the model-level (layer-2) limit is exceeded."""

    def __init__(self, model: str, *, request_id: str | None = None) -> None:
        self.model = model
        super().__init__(
            f"Model '{model}' is currently throttled — too many concurrent or per-minute requests.",
            request_id=request_id,
        )
