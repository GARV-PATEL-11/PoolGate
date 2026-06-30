"""Gemini structured output and invoke examples — sync and async.

Prerequisites:
    pip install poolgate[gemini]

    export TOTAL_GEMINI_KEYS=1
    export GEMINI_API_KEY_01=your-google-ai-studio-key
"""

from __future__ import annotations

import asyncio
from typing import Optional

from pydantic import BaseModel, Field

from poolgate.schemas.common.runtime import RequestConfig
from poolgate.services.gemini_provider import GeminiService

# ---------------------------------------------------------------------------
# Shared schemas used across examples
# ---------------------------------------------------------------------------


class MovieReview(BaseModel):
    title: str
    year: int
    rating: float = Field(ge=0.0, le=10.0)
    summary: str
    pros: list[str]
    cons: list[str]
    recommended: bool


class Address(BaseModel):
    street: str
    city: str
    country: str
    postal_code: str


class Person(BaseModel):
    name: str
    age: int
    occupation: str
    address: Address
    hobbies: list[str]


class TranslationResult(BaseModel):
    original: str
    translated: str
    language_detected: str
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# invoke — sync
# ---------------------------------------------------------------------------


def invoke_simple() -> None:
    """invoke() with just a prompt — the most minimal usage."""
    service = GeminiService()

    response = service.invoke(
        prompt="Explain what a neural network is in one sentence.",
        model="gemini-2.5-flash",
    )
    print(f"[invoke_simple]\n  {response.text}")
    print(f"  tokens={response.usage.total_tokens}  latency={response.latency:.2f}s\n")


def invoke_with_system() -> None:
    """invoke() with a system prompt to control the persona."""
    service = GeminiService()

    response = service.invoke(
        prompt="What should I eat for breakfast?",
        model="gemini-2.5-flash",
        system="You are a professional nutritionist. Keep answers under 40 words.",
    )
    print(f"[invoke_with_system]\n  {response.text}\n")


def invoke_with_config() -> None:
    """invoke() with a custom RequestConfig for temperature and retries."""
    service = GeminiService()

    cfg = RequestConfig(temperature=0.2, max_tokens=80, retries=2)
    response = service.invoke(
        prompt="Give me three words that describe the ocean.",
        model="gemini-2.5-flash",
        config=cfg,
    )
    print(f"[invoke_with_config]\n  {response.text}")
    print(f"  finish_reason={response.finish_reason.value}\n")


def invoke_with_session() -> None:
    """invoke() pinned to a session so multiple calls share context tracking."""
    service = GeminiService()
    sid = "demo-session-42"

    r1 = service.invoke(
        prompt="What is the tallest mountain on Earth?",
        model="gemini-2.5-flash",
        session_id=sid,
    )
    r2 = service.invoke(
        prompt="And the second tallest?",
        model="gemini-2.5-flash",
        session_id=sid,
    )
    print(f"[invoke_with_session]")
    print(f"  Q1: {r1.text.strip()}")
    print(f"  Q2: {r2.text.strip()}")
    print(f"  Both used session_id={r1.session_id}\n")


# ---------------------------------------------------------------------------
# invoke — async
# ---------------------------------------------------------------------------


async def async_invoke_simple() -> None:
    """async_invoke() — awaitable single call."""
    service = GeminiService()

    response = await service.async_invoke(
        prompt="What is the speed of light in km/s?",
        model="gemini-2.5-flash",
    )
    print(f"[async_invoke_simple]\n  {response.text.strip()}\n")


async def async_invoke_parallel() -> None:
    """Fire multiple async_invoke() calls concurrently with asyncio.gather."""
    service = GeminiService()

    prompts = [
        "Capital of Japan?",
        "Capital of Brazil?",
        "Capital of Egypt?",
    ]

    responses = await asyncio.gather(
        *[
            service.async_invoke(
                prompt=p,
                model="gemini-2.5-flash",
                system="Answer in five words or fewer.",
            )
            for p in prompts
        ]
    )

    print("[async_invoke_parallel]")
    for prompt, resp in zip(prompts, responses):
        print(f"  {prompt!r:35s} → {resp.text.strip()}")
    print()


async def async_invoke_with_system_and_config() -> None:
    """async_invoke() combining system prompt, config, and explicit session."""
    service = GeminiService()

    cfg = RequestConfig(temperature=0.5, max_tokens=120)
    response = await service.async_invoke(
        prompt="Summarise the French Revolution.",
        model="gemini-2.5-flash",
        system="You are a historian. Be concise and factual.",
        config=cfg,
        session_id="history-session",
    )
    print(f"[async_invoke_with_system_and_config]")
    print(f"  {response.text.strip()}")
    print(f"  key={response.api_key_id}  tokens={response.usage.total_tokens}\n")


# ---------------------------------------------------------------------------
# structured — sync
# ---------------------------------------------------------------------------


def structured_simple() -> None:
    """structured() returns a validated Pydantic model directly."""
    service = GeminiService()

    review = service.structured(
        prompt="Write a review of the movie Inception (2010).",
        schema=MovieReview,
        model="gemini-2.5-flash",
    )
    print(f"[structured_simple]")
    print(f"  {review.title} ({review.year}) — {review.rating}/10")
    print(f"  Recommended: {review.recommended}")
    print(f"  Summary: {review.summary[:80]}...")
    print(f"  Pros: {review.pros}")
    print(f"  Cons: {review.cons}\n")


def structured_with_system() -> None:
    """structured() with a system prompt to guide tone or language."""
    service = GeminiService()

    result = service.structured(
        prompt="Translate 'Hello, how are you?' into Spanish.",
        schema=TranslationResult,
        model="gemini-2.5-flash",
        system="You are a professional translator. Detect the source language automatically.",
    )
    print(f"[structured_with_system]")
    print(f"  Original:    {result.original}")
    print(f"  Translated:  {result.translated}")
    print(f"  Detected:    {result.language_detected}  (confidence={result.confidence:.0%})\n")


def structured_nested() -> None:
    """structured() with a nested Pydantic schema (Person contains Address)."""
    service = GeminiService()

    person = service.structured(
        prompt="Invent a fictional person named Alice who lives in London and works as an architect.",
        schema=Person,
        model="gemini-2.5-flash",
    )
    print(f"[structured_nested]")
    print(f"  {person.name}, {person.age}, {person.occupation}")
    print(f"  Lives at: {person.address.street}, {person.address.city}, {person.address.country}")
    print(f"  Hobbies: {', '.join(person.hobbies)}\n")


def structured_with_config_and_retries() -> None:
    """structured() with low temperature for determinism and extra json_retries."""
    service = GeminiService()

    cfg = RequestConfig(temperature=0.1, max_tokens=512)
    review = service.structured(
        prompt="Review the film 2001: A Space Odyssey.",
        schema=MovieReview,
        model="gemini-2.5-flash",
        config=cfg,
        json_retries=3,  # retry JSON repair up to 3 times before giving up
    )
    print(f"[structured_with_config_and_retries]")
    print(f"  {review.title} — rating {review.rating}/10")
    print(f"  {review.summary[:100]}...\n")


# ---------------------------------------------------------------------------
# structured — async
# ---------------------------------------------------------------------------


async def async_structured_simple() -> None:
    """async_structured() — awaitable structured call."""
    service = GeminiService()

    review = await service.async_structured(
        prompt="Review the movie The Matrix (1999).",
        schema=MovieReview,
        model="gemini-2.5-flash",
    )
    print(f"[async_structured_simple]")
    print(f"  {review.title} — {review.rating}/10  recommended={review.recommended}\n")


async def async_structured_parallel() -> None:
    """Run several async_structured() calls concurrently — one schema, many prompts."""
    service = GeminiService()

    prompts = [
        ("Interstellar (2014)", MovieReview),
        ("Parasite (2019)", MovieReview),
        ("Everything Everywhere All at Once (2022)", MovieReview),
    ]

    results = await asyncio.gather(
        *[
            service.async_structured(
                prompt=f"Review the movie {title}.",
                schema=schema,
                model="gemini-2.5-flash",
                system="Keep the summary under 20 words.",
            )
            for title, schema in prompts
        ]
    )

    print("[async_structured_parallel]")
    for review in results:
        assert isinstance(review, MovieReview)
        bar = "█" * int(review.rating) + "░" * (10 - int(review.rating))
        print(f"  {review.title:<45s} [{bar}] {review.rating}/10")
    print()


async def async_structured_different_schemas() -> None:
    """Parallel async_structured() with different schema types in the same gather."""
    service = GeminiService()

    review_task = service.async_structured(
        prompt="Review the movie Dune (2021).",
        schema=MovieReview,
        model="gemini-2.5-flash",
    )
    translation_task = service.async_structured(
        prompt="Translate 'Good morning, have a great day!' into French.",
        schema=TranslationResult,
        model="gemini-2.5-flash",
    )
    person_task = service.async_structured(
        prompt="Invent a fictional scientist named Bob who lives in Berlin.",
        schema=Person,
        model="gemini-2.5-flash",
    )

    review, translation, person = await asyncio.gather(review_task, translation_task, person_task)

    print("[async_structured_different_schemas]")
    print(f"  Movie:       {review.title} — {review.rating}/10")  # type: ignore[union-attr]
    print(f"  Translation: {translation.translated!r}")  # type: ignore[union-attr]
    print(f"  Person:      {person.name}, {person.occupation}, {person.address.city}")  # type: ignore[union-attr]
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("INVOKE — SYNC")
    print("=" * 60)
    invoke_simple()
    invoke_with_system()
    invoke_with_config()
    invoke_with_session()

    print("=" * 60)
    print("INVOKE — ASYNC")
    print("=" * 60)
    asyncio.run(async_invoke_simple())
    asyncio.run(async_invoke_parallel())
    asyncio.run(async_invoke_with_system_and_config())

    print("=" * 60)
    print("STRUCTURED — SYNC")
    print("=" * 60)
    structured_simple()
    structured_with_system()
    structured_nested()
    structured_with_config_and_retries()

    print("=" * 60)
    print("STRUCTURED — ASYNC")
    print("=" * 60)
    asyncio.run(async_structured_simple())
    asyncio.run(async_structured_parallel())
    asyncio.run(async_structured_different_schemas())
