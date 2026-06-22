"""Async chat — async_invoke and async_stream with asyncio."""

from __future__ import annotations

import asyncio
import sys

from exceptions.base import GroqServiceError
from services.provider_service import GroqService
from dotenv import load_dotenv


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


if __name__ == "__main__":
	try:
		asyncio.run(main())
	except GroqServiceError as e:
		print(f"\nError: {e}", file=sys.stderr)
		sys.exit(1)
