"""Streaming chat — print tokens as they arrive.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
    POOLGATE_DATA_DIR=./poolgate_data
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from exceptions import GroqServiceError
from services.provider_service import GroqService

load_dotenv()
os.environ.setdefault("POOLGATE_DATA_DIR", "./poolgate_data")


def main() -> None:
    service = GroqService()

    print("Streaming response: ", end="", flush=True)
    for chunk in service.stream(
        "Write a short poem about the ocean.",
        model="llama-3.3-70b-versatile",
        system="You are a poet.",
    ):
        print(chunk, end="", flush=True)
    print()

    health = service.health()
    print(f"\nPool health: {health.status}, active keys: {health.active_keys}")

    service.flush_tracking()
    if service._config.data_dir:
        print(f"Data saved to {service._config.data_dir}/")


if __name__ == "__main__":
    try:
        main()
    except GroqServiceError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
