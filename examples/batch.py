"""Batch processing — run multiple prompts concurrently and collect results."""

from __future__ import annotations

import asyncio

from exceptions.base import GroqServiceError
from services.provider_service import GroqService
from dotenv import load_dotenv


load_dotenv()
PROMPTS = [
	"What is 2 + 2?",
	"Name the planets in our solar system.",
	"Write a haiku about Python.",
	"What is the boiling point of water in Celsius?",
	"Who wrote Romeo and Juliet?",
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
		f"Batch complete: {summary.succeeded}/{summary.total} succeeded  ({summary.failed} failed)",
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


if __name__ == "__main__":
	try:
		asyncio.run(main())
	except GroqServiceError as e:
		print(f"Error: {e}")
