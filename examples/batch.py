"""Batch processing — run multiple prompts concurrently and collect results.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=2          # more keys = higher throughput
    GROQ_API_KEY_01=gsk_...
    GROQ_API_KEY_02=gsk_...
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from exceptions.base import GroqServiceError
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


async def main() -> None:
    service = GroqService()

    summary = await service.batch(
        prompts=PROMPTS,
        model="llama-3.3-70b-versatile",
        system="Answer as briefly as possible.",
        concurrency=3,
    )

    print(
        f"Batch complete: {summary.succeeded}/{summary.total} succeeded  "
        f"({summary.failed} failed)",
    )
    print(f"Total latency: {summary.total_latency:.3f}s")
    print()

    for result in summary.results:
        status = "OK" if result.success else "FAIL"
        if result.success and result.response:
            preview = result.response.text[:60].replace("\n", " ")
            print(f"  [{status}] #{result.index}: {preview}...")
        else:
            print(f"  [{status}] #{result.index}: {result.error}")

    service.flush_tracking()
    if service._config.paths.base_dir:
        print(
            f"\nAll {summary.total} requests journaled to {service._config.paths.base_dir}/requests/"
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except GroqServiceError as e:
        print(f"Error: {e}")
