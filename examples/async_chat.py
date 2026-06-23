"""Async chat — async_invoke and async_stream with asyncio.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from exceptions.base import GroqServiceError
from services.provider_service import GroqService


load_dotenv()


async def main() -> None:
	service = GroqService()

	# Single async call
	response = await service.async_invoke(
		"Name three programming languages in one sentence.",
		model="llama-3.3-70b-versatile",
		)
	print(f"Async invoke: {response.text}")
	print(f"Latency: {response.latency:.3f}s")

	# Async streaming
	print("\nAsync stream: ", end="", flush=True)
	async for chunk in service.async_stream(
			"Explain recursion in two sentences.",
			model="llama-3.3-70b-versatile",
			):
		print(chunk, end="", flush=True)
	print()

	service.flush_tracking()
	if service._config.paths.base_dir:
		print(f"\nData saved to {service._config.paths.base_dir}/")


if __name__ == "__main__":
	try:
		asyncio.run(main())
	except GroqServiceError as e:
		print(f"\nError: {e}", file=sys.stderr)
		sys.exit(1)
