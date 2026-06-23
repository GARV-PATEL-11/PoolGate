"""Two-layer throttling — configure per-capability and per-model rate limits.

PoolGate's throttle system has two independent layers that stack:

  Layer 1 (capability): applies to every request of a given type
    (chat, structured output, moderation, transcription, synthesis).
    Uses a sliding-window counter so limits are exact.

  Layer 2 (model): applies per-model, independently of capability.
    Uses a token bucket so the rate is smooth. Only active for
    models you explicitly add to model_configs.

ThrottleMiddleware is created automatically inside GroqService with
sensible defaults (25 RPM / 10 concurrent for chat). You can replace it
by passing a custom ThrottleConfig.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
"""

from __future__ import annotations

import time

from dotenv import load_dotenv

from exceptions import CapabilityThrottledError, ModelThrottledError
from exceptions.base import GroqServiceError
from services.provider_service import GroqService
from throttling import (
    CapabilityConfig,
    CapabilityType,
    ModelConfig,
    ThrottleConfig,
    ThrottleMiddleware,
    ThrottleMode,
)

load_dotenv()


# ── 1. Default throttle config ────────────────────────────────────────────────


def demo_defaults() -> None:
    """Show the built-in defaults that activate when you create GroqService()."""
    print("=== Default Throttle Config ===")
    cfg = ThrottleConfig()
    for cap_type, cap_cfg in cfg.capability_configs.items():
        mode = cap_cfg.mode.value
        print(
            f"  {cap_type.value:<20}  rpm={cap_cfg.max_rpm:<4}" f"  concurrent={cap_cfg.max_concurrent:<3}  mode={mode}"
        )
    print()


# ── 2. Capability-level throttle ──────────────────────────────────────────────


def demo_capability_throttle() -> None:
    """
    Layer 1: limit how many chat requests run per minute / concurrently.
    Here we set a very tight limit (2 RPM) to trigger the error intentionally.
    """
    print("=== Layer 1: Capability Throttle ===")

    cfg = ThrottleConfig(
        capability_configs={
            CapabilityType.TEXT_GENERATION: CapabilityConfig(
                max_rpm=2,
                max_concurrent=5,
            ),
        }
    )
    throttle = ThrottleMiddleware(cfg)

    # Two requests pass fine
    for i in range(1, 3):
        h = throttle.check("chat", "llama-3.3-70b-versatile", f"req-{i}")
        print(f"  Request {i}: allowed")
        h.release()

    # Third request hits the RPM ceiling
    try:
        throttle.check("chat", "llama-3.3-70b-versatile", "req-3")
    except CapabilityThrottledError as exc:
        print(f"  Request 3: BLOCKED — {exc}")

    print()


# ── 3. Model-level throttle ───────────────────────────────────────────────────


def demo_model_throttle() -> None:
    """
    Layer 2: limit requests to a specific model, regardless of capability.
    Useful when you want the heavy 70B model capped but lighter models free.
    """
    print("=== Layer 2: Model Throttle ===")

    cfg = ThrottleConfig(
        model_configs={
            "llama-3.3-70b-versatile": ModelConfig(
                max_rpm=100,
                max_concurrent=1,  # only 1 concurrent request to this model
            ),
        }
    )
    throttle = ThrottleMiddleware(cfg)

    # First request acquires the only concurrent slot
    h1 = throttle.check("chat", "llama-3.3-70b-versatile", "req-1")
    print("  Request 1 (70B): allowed")

    # Second concurrent request is blocked at the model layer
    try:
        throttle.check("chat", "llama-3.3-70b-versatile", "req-2")
    except ModelThrottledError as exc:
        print(f"  Request 2 (70B): BLOCKED — {exc}")

    # A different model is completely unaffected
    h3 = throttle.check("chat", "llama-3.1-8b-instant", "req-3")
    print("  Request 3 (8B):  allowed (different model)")

    h1.release()
    h3.release()
    print()


# ── 4. Combined config ────────────────────────────────────────────────────────


def demo_combined_config() -> None:
    """
    Production-style config: per-capability limits PLUS specific model caps.
    Both layers must pass for a request to proceed.
    """
    print("=== Combined Config (L1 + L2) ===")

    cfg = ThrottleConfig(
        capability_configs={
            CapabilityType.TEXT_GENERATION: CapabilityConfig(
                max_rpm=30,
                max_concurrent=10,
                burst_multiplier=1.2,  # effective RPM = 36 (allows short bursts)
            ),
            CapabilityType.MODERATION: CapabilityConfig(
                max_rpm=60,
                max_concurrent=20,
            ),
            CapabilityType.TRANSCRIPTION: CapabilityConfig(
                max_rpm=10,
                max_concurrent=4,
                mode=ThrottleMode.CONCURRENCY_ONLY,  # audio: ignore RPM, cap concurrency
            ),
        },
        model_configs={
            "llama-3.3-70b-versatile": ModelConfig(max_rpm=20, max_concurrent=5),
            "meta-llama/llama-prompt-guard-2-86m": ModelConfig(max_rpm=60, max_concurrent=15),
        },
    )

    print(f"  Throttle enabled: {cfg.enabled}")
    print(f"  Capability types configured: {len(cfg.capability_configs)}")
    print(f"  Models with explicit limits: {list(cfg.model_configs)}")
    print()

    throttle = ThrottleMiddleware(cfg)

    # Chat request — goes through both L1 (TEXT_GENERATION) and L2 (70B model)
    h = throttle.check("chat", "llama-3.3-70b-versatile", "chat-req")
    print("  chat / 70B:            allowed")
    h.release()

    # Moderation — L1 (MODERATION) + L2 (guard model)
    h = throttle.check("moderation", "meta-llama/llama-prompt-guard-2-86m", "mod-req")
    print("  moderation / guard:    allowed")
    h.release()

    # Transcription — L1 only (no model config for whisper), concurrency-only mode
    h = throttle.check("transcription", "whisper-large-v3", "trans-req")
    print("  transcription / whisper: allowed (concurrency-only mode)")
    h.release()

    print()


# ── 5. Disabled throttle ──────────────────────────────────────────────────────


def demo_disabled() -> None:
    """
    Pass ThrottleConfig(enabled=False) to bypass all limits entirely.
    Useful during development or integration testing.
    """
    print("=== Disabled Throttle ===")

    cfg = ThrottleConfig(enabled=False)
    throttle = ThrottleMiddleware(cfg)

    for i in range(10):
        h = throttle.check("chat", "any-model", f"req-{i}")
        h.release()

    print("  10 requests, no limits enforced")
    print()


# ── 6. RAII handle pattern ────────────────────────────────────────────────────


def demo_raii_handle() -> None:
    """
    ThrottleHandle must be released in a finally block.
    The handle decrements concurrency counters so the slot is freed even if
    the downstream API call raises an exception.
    """
    print("=== RAII Handle Pattern ===")

    cfg = ThrottleConfig(
        capability_configs={CapabilityType.TEXT_GENERATION: CapabilityConfig(max_rpm=100, max_concurrent=2)}
    )
    throttle = ThrottleMiddleware(cfg)

    def call_with_cleanup(request_id: str, fail: bool = False) -> str:
        handle = throttle.check("chat", "llama-3.3-70b-versatile", request_id)
        try:
            if fail:
                raise RuntimeError("simulated upstream failure")
            return f"ok({request_id})"
        finally:
            # Always release — whether the call succeeded or raised
            handle.release()

    result = call_with_cleanup("r1")
    print(f"  r1 result: {result}")

    try:
        call_with_cleanup("r2", fail=True)
    except RuntimeError:
        pass

    # Slot was released by the finally block — r3 is allowed
    result = call_with_cleanup("r3")
    print(f"  r3 result: {result}  (slot freed by finally even after r2 failure)")
    print()


# ── 7. Wire into GroqService ──────────────────────────────────────────────────


def demo_with_service() -> None:
    """
    Pass throttle_config to GroqService to replace the default throttle.
    All invoke/stream/batch calls go through the custom limits automatically.
    """
    print("=== GroqService with Custom Throttle ===")

    cfg = ThrottleConfig(
        capability_configs={
            CapabilityType.TEXT_GENERATION: CapabilityConfig(
                max_rpm=25,
                max_concurrent=10,
                burst_multiplier=1.2,
            ),
            CapabilityType.STRUCTURED_GEN: CapabilityConfig(
                max_rpm=15,
                max_concurrent=5,
            ),
            CapabilityType.MODERATION: CapabilityConfig(
                max_rpm=60,
                max_concurrent=20,
            ),
            CapabilityType.TRANSCRIPTION: CapabilityConfig(
                max_rpm=10,
                max_concurrent=4,
                mode=ThrottleMode.CONCURRENCY_ONLY,
            ),
            CapabilityType.SYNTHESIS: CapabilityConfig(
                max_rpm=5,
                max_concurrent=3,
                mode=ThrottleMode.CONCURRENCY_ONLY,
            ),
        },
        model_configs={
            "llama-3.3-70b-versatile": ModelConfig(max_rpm=20, max_concurrent=8),
        },
    )

    service = GroqService(throttle_config=cfg)

    start = time.perf_counter()
    response = service.invoke(
        "Name one planet in our solar system.",
        model="llama-3.3-70b-versatile",
    )
    elapsed = time.perf_counter() - start

    print(f"  Answer:  {response.text.strip()}")
    print(f"  Tokens:  {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")
    print(f"  Latency: {elapsed:.3f}s")
    print()


def main() -> None:
    demo_defaults()
    demo_capability_throttle()
    demo_model_throttle()
    demo_combined_config()
    demo_disabled()
    demo_raii_handle()
    demo_with_service()


if __name__ == "__main__":
    try:
        main()
    except GroqServiceError as e:
        print(f"GroqServiceError: {e}")
