"""ThrottleHandle and ThrottleMiddleware — the public face of the throttle system."""

from __future__ import annotations

from exceptions.throttle import CapabilityThrottledError, ModelThrottledError
from throttling.config import ThrottleConfig
from throttling.layer1 import CapabilityThrottle
from throttling.layer2 import ModelThrottle
from throttling.types import CapabilityType

# Maps provider_service.py capability strings to CapabilityType enum values.
_CAP_MAP: dict[str, CapabilityType] = {
    "chat": CapabilityType.TEXT_GENERATION,
    "structured": CapabilityType.STRUCTURED_GEN,
    "tools": CapabilityType.TOOL_CALLING,
    "moderation": CapabilityType.MODERATION,
    "transcription": CapabilityType.TRANSCRIPTION,
    "translation": CapabilityType.TRANSCRIPTION,
    "synthesis": CapabilityType.SYNTHESIS,
    "api": CapabilityType.TEXT_GENERATION,
}


class ThrottleHandle:
    """
    RAII handle returned by ThrottleMiddleware.check().

    Always call release() in a finally block — it decrements concurrency
    counters for both layers and is idempotent (safe to call multiple times).
    """

    __slots__ = ("_l1", "_l1_cap", "_l2", "_l2_model", "_released")

    def __init__(
        self,
        *,
        l1: CapabilityThrottle | None,
        l1_cap: CapabilityType | None,
        l2: ModelThrottle | None,
        l2_model: str | None,
    ) -> None:
        self._l1 = l1
        self._l1_cap = l1_cap
        self._l2 = l2
        self._l2_model = l2_model
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._l2 is not None and self._l2_model is not None:
            self._l2.release_concurrent(self._l2_model)
        if self._l1 is not None and self._l1_cap is not None:
            self._l1.release_concurrent(self._l1_cap)


class ThrottleMiddleware:
    """
    Two-layer in-memory throttle for GroqService.

    Layer 1 — capability (e.g. "chat", "moderation"): sliding-window RPM
    + concurrency, shared across all models of that capability type.

    Layer 2 — model (e.g. "llama-3.3-70b-versatile"): token-bucket RPM
    + concurrency. Only active for models explicitly in ThrottleConfig.model_configs.

    Usage::

        middleware = ThrottleMiddleware(config)
        handle = middleware.check(capability, model, request_id)
        try:
            # ... make the actual API call ...
        finally:
            handle.release()
    """

    def __init__(self, config: ThrottleConfig | None = None) -> None:
        self._config = config or ThrottleConfig()
        self._l1 = CapabilityThrottle(self._config)
        self._l2 = ModelThrottle(self._config)

    def check(self, capability: str, model: str, request_id: str) -> ThrottleHandle:
        """
        Check both throttle layers.

        Raises CapabilityThrottledError if the capability layer is saturated.
        Raises ModelThrottledError if the model layer is saturated.
        Returns a ThrottleHandle whose release() must be called in a finally block.
        """
        if not self._config.enabled:
            return ThrottleHandle(l1=None, l1_cap=None, l2=None, l2_model=None)

        cap_type = _CAP_MAP.get(capability, CapabilityType.TEXT_GENERATION)

        # Layer 1: RPM check (sliding window — timestamp expires, no rollback needed on failure)
        if not self._l1.try_acquire_rpm(cap_type):
            raise CapabilityThrottledError(capability, request_id=request_id)

        # Layer 1: concurrency check (must be released if layer 2 fails below)
        if not self._l1.try_acquire_concurrent(cap_type):
            raise CapabilityThrottledError(capability, request_id=request_id)

        # Layer 2: model checks (only when model has explicit config)
        l2_active = False
        if self._l2.has_config(model):
            if not self._l2.try_acquire_rpm(model):
                self._l1.release_concurrent(cap_type)
                raise ModelThrottledError(model, request_id=request_id)

            if not self._l2.try_acquire_concurrent(model):
                self._l1.release_concurrent(cap_type)
                raise ModelThrottledError(model, request_id=request_id)

            l2_active = True

        return ThrottleHandle(
            l1=self._l1,
            l1_cap=cap_type,
            l2=self._l2 if l2_active else None,
            l2_model=model if l2_active else None,
        )
