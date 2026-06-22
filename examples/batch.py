"""Batch processing — run multiple prompts concurrently and collect results.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=2          # more keys = higher throughput
    GROQ_API_KEY_01=gsk_...
    GROQ_API_KEY_02=gsk_...
    POOLGATE_DATA_DIR=./poolgate_data
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from exceptions.base import GroqServiceError
from services.provider_service import GroqService


load_dotenv()
os.environ.setdefault("POOLGATE_DATA_DIR", "./poolgate_data")

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
	if service._config.data_dir:
		print(f"\nAll {summary.total} requests journaled to {service._config.data_dir}/requests/")


if __name__ == "__main__":
	try:
		asyncio.run(main())
	except GroqServiceError as e:
		print(f"Error: {e}")
