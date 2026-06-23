"""Error handling — catch specific PoolGate exception types.

Errors are written to logs/error.log inside the configured data directory.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from exceptions import (
    APIKeyDisabledError,
    GroqServiceError,
    InvalidRequestError,
    NoAvailableAPIKeyError,
    RateLimitExceededError,
)
from services.provider_service import GroqService

load_dotenv()

PROMPTS = [
    "What is 2 + 2?",
    "Name the planets in our solar system.",
    "Write a haiku about Python.",
    "What is the boiling point of water in Celsius?",
    "Who wrote Romeo and Juliet?",
    "What is the capital of France?",
    "Explain photosynthesis in simple terms.",
    "What is the speed of light?",
    "Define gravity.",
    "Translate 'good morning' into Spanish.",
    "What is the square root of 144?",
    "List the continents of the world.",
    "Who discovered gravity?",
    "What is AI?",
    "Write a short joke about programmers.",
    "What is the largest ocean on Earth?",
    "Explain recursion in programming.",
    "What is HTTP?",
    "What is 10 factorial?",
    "Name three programming languages.",
    "What is a database?",
    "What is machine learning?",
    "Convert 100 Celsius to Fahrenheit.",
    "What is the tallest mountain in the world?",
    "Who is Albert Einstein?",
    "What is an API?",
    "Explain JSON in one sentence.",
    "What is 5 * 6?",
    "What is the meaning of life (philosophically)?",
    "Write a Python one-liner to reverse a string.",
    "What is Docker?",
    "What is Git used for?",
    "Explain cloud computing briefly.",
    "What is cybersecurity?",
    "Name 5 animals that live in the ocean.",
    "What is the Pythagorean theorem?",
]


def safe_invoke(service: GroqService, prompt: str) -> str | None:
    try:
        response = service.invoke(prompt, model="llama-3.3-70b-versatile")
        return response.text

    except NoAvailableAPIKeyError as e:
        print(f"[NoAvailableAPIKeyError] Pool exhausted: {e.total_keys} keys")
        print(f"  Reason breakdown: {e.reason_counts}")
        return None

    except APIKeyDisabledError as e:
        print(
            f"[APIKeyDisabledError] Key {e.key_id!r} is disabled (HTTP {e.status_code})"
        )
        return None

    except RateLimitExceededError as e:
        retry_msg = f", retry after {e.retry_after}s" if e.retry_after else ""
        print(f"[RateLimitExceededError] Rate limited{retry_msg}")
        return None

    except InvalidRequestError as e:
        print(f"[InvalidRequestError] Bad request: {e}")
        return None

    except GroqServiceError as e:
        print(f"[GroqServiceError] {type(e).__name__}: {e}")
        return None


def main() -> None:
    service = GroqService()

    # -------------------------
    # BATCH PROMPT EXECUTION
    # -------------------------
    print("\n=== SAFE INVOKE BATCH ===")
    for i, prompt in enumerate(PROMPTS, 1):
        result = safe_invoke(service, prompt)
        if result:
            print(f"[{i}] {prompt}")
            print(f"    → Success: {result.strip()}")
        else:
            print(f"[{i}] {prompt}")
            print("    → Failed (handled safely)")

    # -------------------------
    # INVALID ROLE TEST
    # -------------------------
    print("\n=== INVALID ROLE TEST ===")
    try:
        service.chat(
            messages=[{"role": "narrator", "content": "Tell the story."}],
            model="llama-3.3-70b-versatile",
        )
    except InvalidRequestError as e:
        print(f"Expected error for invalid role: {type(e).__name__}: {e}")

    # -------------------------
    # HEALTH + STATS
    # -------------------------
    health = service.health()
    stats = service.get_global_stats()

    print(f"\nPool: {health.status}, active keys: {health.active_keys}")
    print(
        f"Requests: {stats['total_requests']} total, "
        f"{stats['successful_requests']} succeeded, "
        f"{stats['failed_requests']} failed",
    )

    service.flush_tracking()

    if service._config.paths.log_dir:
        print(f"\nLogs written to {service._config.paths.log_dir}/")
        print("  general.log — all messages")
        print("  error.log   — errors only")
        print("  request.log — request lifecycle events")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
