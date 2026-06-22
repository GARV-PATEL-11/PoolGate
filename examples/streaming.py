"""Streaming chat — print tokens as they arrive."""

from __future__ import annotations

import sys

from exceptions import GroqServiceError
from services.provider_service import GroqService
from dotenv import load_dotenv


load_dotenv()


def main() -> None:
	service = GroqService()

	print("Streaming response: ", end="", flush=True)
	for chunk in service.stream(
			"Write a short poem about the ocean.",
			model="llama-3.3-70b-versatile",
			system="You are a poet.",
			):
		print(chunk, end="", flush=True)
	print()  # newline after stream ends

	health = service.health()
	print(f"\nPool health: {health.status}, active keys: {health.active_keys}")


if __name__ == "__main__":
	try:
		main()
	except GroqServiceError as e:
		print(f"\nError: {e}", file=sys.stderr)
		sys.exit(1)
