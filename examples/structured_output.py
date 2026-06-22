"""Structured output — get a validated Pydantic model back from the LLM.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
    POOLGATE_DATA_DIR=./poolgate_data
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from exceptions import GroqServiceError, StructuredOutputError
from services.provider_service import GroqService

load_dotenv()
os.environ.setdefault("POOLGATE_DATA_DIR", "./poolgate_data")


class MovieReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    rating: float  # 0.0–10.0
    summary: str
    recommended: bool


def main() -> None:
    service = GroqService()

    review: MovieReview = service.structured(
        "Review the movie 'Inception' by Christopher Nolan.",
        schema=MovieReview,
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        system="You are a film critic. Respond with a structured review.",
    )
    print(f"Title:       {review.title}")
    print(f"Rating:      {review.rating}/10")
    print(f"Recommended: {review.recommended}")
    print(f"Summary:     {review.summary}")

    service.flush_tracking()
    if service._config.data_dir:
        print(f"\nData saved to {service._config.data_dir}/")


if __name__ == "__main__":
    try:
        main()
    except StructuredOutputError as e:
        print(f"Structured output failed: {e}")
    except GroqServiceError as e:
        print(f"Error: {e}")
