#!/usr/bin/env python3
"""
scripts/smoke_test.py

The cheapest possible verification that the package is in a working state:
import it, and construct a GroqService with a dummy key. No network calls
are made. Run this before anything else (it takes well under a second) —
it is specifically designed to catch the class of bug found in the original
audit, where the package imported "successfully" at the module level but
GroqService() itself raised TypeError on construction because one of its
internal clients (ChatClient) was an abstract class with unimplemented
methods.

Usage:
    python scripts/smoke_test.py

Exit code 0 on success, 1 on failure (with a clear error printed either way).
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("TOTAL_GROQ_KEYS", "1")
    os.environ.setdefault("GROQ_API_KEY_01", "gsk_smoke_test_dummy_key")

    try:
        from poolgate.services.provider import GroqService
    except Exception as exc:
        print(f"FAIL: could not import poolgate.services.provider: {type(exc).__name__}: {exc}")
        return 1

    try:
        service = GroqService()
    except Exception as exc:
        print(f"FAIL: GroqService() raised on construction: {type(exc).__name__}: {exc}")
        return 1

    required_methods = (
        "invoke",
        "async_invoke",
        "chat",
        "async_chat",
        "structured",
        "async_structured",
        "stream",
        "async_stream",
        "batch",
        "health",
        "get_key_pool_status",
        "get_global_stats",
    )
    missing = [name for name in required_methods if not hasattr(service, name)]
    if missing:
        print(f"FAIL: GroqService is missing expected methods: {missing}")
        return 1

    health = service.health()
    if health.active_keys < 1:
        print("FAIL: GroqService constructed but reports zero active keys.")
        return 1

    print("OK: GroqService imports, constructs, and reports a healthy key pool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
