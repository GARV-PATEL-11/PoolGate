"""
Benchmark structured() vs async_structured() using 100 prompts.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
"""

from __future__ import annotations

import asyncio
import time

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from exceptions import GroqServiceError, StructuredOutputError
from services.provider_service import GroqService

load_dotenv()


class MovieReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    rating: float
    summary: str
    recommended: bool


PROMPTS = [
    "Review the movie 'Inception' by Christopher Nolan.",
    "Review the movie 'Interstellar' by Christopher Nolan.",
    "Review the movie 'The Dark Knight'.",
    "Review the movie 'Memento'.",
    "Review the movie 'Tenet'.",
    "Review the movie 'Oppenheimer'.",
    "Review the movie 'The Prestige'.",
    "Review the movie 'Dunkirk'.",
    "Review the movie 'Batman Begins'.",
    "Review the movie 'Insomnia'.",
    "Review the movie 'Avatar'.",
    "Review the movie 'Avatar: The Way of Water'.",
    "Review the movie 'Titanic'.",
    "Review the movie 'The Terminator'.",
    "Review the movie 'Terminator 2: Judgment Day'.",
    "Review the movie 'Aliens'.",
    "Review the movie 'The Abyss'.",
    "Review the movie 'True Lies'.",
    "Review the movie 'Pulp Fiction'.",
    "Review the movie 'Reservoir Dogs'.",
    "Review the movie 'Kill Bill Vol. 1'.",
    "Review the movie 'Kill Bill Vol. 2'.",
    "Review the movie 'Django Unchained'.",
    "Review the movie 'Inglourious Basterds'.",
    "Review the movie 'Once Upon a Time in Hollywood'.",
    "Review the movie 'The Hateful Eight'.",
    "Review the movie 'Fight Club'.",
    "Review the movie 'Se7en'.",
    "Review the movie 'Gone Girl'.",
    "Review the movie 'The Social Network'.",
    "Review the movie 'Zodiac'.",
    "Review the movie 'The Curious Case of Benjamin Button'.",
    "Review the movie 'The Matrix'.",
    "Review the movie 'The Matrix Reloaded'.",
    "Review the movie 'The Matrix Revolutions'.",
    "Review the movie 'John Wick'.",
    "Review the movie 'John Wick: Chapter 2'.",
    "Review the movie 'John Wick: Chapter 3'.",
    "Review the movie 'John Wick: Chapter 4'.",
    "Review the movie " "'Speed'.",
    "Review the movie 'The Shawshank Redemption'.",
    "Review the movie 'The Green Mile'.",
    "Review the movie 'Forrest Gump'.",
    "Review the movie 'Saving Private Ryan'.",
    "Review the movie 'Catch Me If You Can'.",
    "Review the movie 'The Terminal'.",
    "Review the movie 'Bridge of Spies'.",
    "Review the movie 'Lincoln'.",
    "Review the movie 'Jaws'.",
    "Review the movie 'E.T. the Extra-Terrestrial'.",
    "Review the movie 'Jurassic Park'.",
    "Review the movie 'The Lost World: Jurassic Park'.",
    "Review the movie 'Schindler's List'.",
    "Review the movie 'Ready Player One'.",
    "Review the movie 'Minority Report'.",
    "Review the movie 'War of the Worlds'.",
    "Review the movie 'The Godfather'.",
    "Review the movie 'The Godfather Part II'.",
    "Review the movie 'Goodfellas'.",
    "Review the movie 'Casino'.",
    "Review the movie 'The Irishman'.",
    "Review the movie 'Taxi Driver'.",
    "Review the movie 'Raging Bull'.",
    "Review the movie 'The Departed'.",
    "Review the movie 'Shutter Island'.",
    "Review the movie 'Wolf of Wall " "Street'.",
    "Review the movie 'Gladiator'.",
    "Review the movie 'The Martian'.",
    "Review the movie 'Alien'.",
    "Review the movie 'Blade Runner'.",
    "Review the movie 'Blade Runner 2049'.",
    "Review the movie 'Arrival'.",
    "Review the movie 'Prisoners'.",
    "Review the movie 'Sicario'.",
    "Review the movie 'Dune'.",
    "Review the movie 'Dune: Part Two'.",
    "Review the movie 'No Country for Old Men'.",
    "Review the movie 'Fargo'.",
    "Review the movie 'The Big Lebowski'.",
    "Review the movie 'O Brother, Where Art Thou?'.",
    "Review the movie 'There Will Be Blood'.",
    "Review the movie 'Phantom Thread'.",
    "Review the movie 'Boogie Nights'.",
    "Review the movie 'Magnolia'.",
    "Review the movie 'Whiplash'.",
    "Review the movie 'La La Land'.",
    "Review the movie 'Babylon'.",
    "Review the movie 'Moonlight'.",
    "Review the movie 'Parasite'.",
    "Review the movie 'Oldboy'.",
    "Review the movie 'Memories of Murder'.",
    "Review the movie 'Spirited Away'.",
    "Review the movie 'Princess Mononoke'.",
    "Review the movie 'Howl's Moving Castle'.",
    "Review the movie 'Your Name'.",
    "Review the movie 'Everything Everywhere All at Once'.",
    "Review the movie 'The Grand Budapest Hotel'.",
    "Review the movie 'The French Dispatch'.",
    "Review the movie 'Asteroid City'.",
]


def benchmark_structured(service: GroqService) -> tuple[float, int]:
    print("\nRunning structured() benchmark...")

    success = 0
    start = time.perf_counter()

    for prompt in PROMPTS:
        try:
            service.structured(
                prompt,
                schema=MovieReview,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                system="You are a film critic. Respond with a structured review.",
            )
            success += 1
        except Exception as exc:
            print(f"FAILED: {prompt[:50]}... -> {exc}")

    elapsed = time.perf_counter() - start
    return elapsed, success


async def benchmark_async_structured(service: GroqService) -> tuple[float, int]:
    print("\nRunning async_structured() benchmark...")

    async def run_prompt(prompt: str) -> bool:
        try:
            await service.async_structured(
                prompt,
                schema=MovieReview,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                system="You are a film critic. Respond with a structured review.",
            )
            return True
        except Exception as exc:
            print(f"FAILED: {prompt[:50]}... -> {exc}")
            return False

    start = time.perf_counter()

    results = await asyncio.gather(
        *(run_prompt(prompt) for prompt in PROMPTS),
        return_exceptions=False,
    )

    elapsed = time.perf_counter() - start
    success = sum(results)

    return elapsed, success


def main() -> None:
    service = GroqService()

    structured_time, structured_success = benchmark_structured(service)

    async_time, async_success = asyncio.run(
        benchmark_async_structured(service),
    )

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(
        f"structured() successes: " f"{structured_success}/{len(PROMPTS)}",
    )
    print(
        f"structured() total time: " f"{structured_time:.2f}s",
    )
    print(
        f"structured() avg latency: " f"{structured_time / len(PROMPTS):.3f}s",
    )

    print()

    print(
        f"async_structured() successes: " f"{async_success}/{len(PROMPTS)}",
    )
    print(
        f"async_structured() total time: " f"{async_time:.2f}s",
    )
    print(
        f"async_structured() avg latency: " f"{async_time / len(PROMPTS):.3f}s",
    )

    if async_time > 0:
        print()
        print(
            f"Throughput speedup: " f"{structured_time / async_time:.2f}x",
        )

    health = service.health()

    print()
    print(f"Pool health: {health.status}")
    print(f"Active keys: {health.active_keys}")

    service.flush_tracking()

    if service._config.paths.base_dir:
        print(
            f"Data saved to " f"{service._config.paths.base_dir}/",
        )


if __name__ == "__main__":
    try:
        main()
    except StructuredOutputError as exc:
        print(f"Structured output failed: {exc}")
    except GroqServiceError as exc:
        print(f"Error: {exc}")
