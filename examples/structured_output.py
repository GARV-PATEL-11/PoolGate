"""Structured output — get a validated Pydantic model back from the LLM."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from exceptions import GroqServiceError, StructuredOutputError
from services.provider_service import GroqService
from dotenv import load_dotenv


load_dotenv()


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


if __name__ == "__main__":
	try:
		main()
	except StructuredOutputError as e:
		print(f"Structured output failed: {e}")
	except GroqServiceError as e:
		print(f"Error: {e}")
