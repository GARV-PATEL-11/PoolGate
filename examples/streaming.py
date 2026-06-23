"""
Streaming chat — print tokens as they arrive.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from exceptions import GroqServiceError
from services.provider_service import GroqService


load_dotenv()


def main() -> None:
	service = GroqService()

	prompt = ("Write a long-form, highly detailed technical essay about ocean ecosystems. "
	          "Cover marine biology, deep ocean zones, coral reefs, climate impact, "
	          "case studies, and conservation strategies. "
	          "Target length: ~3000 tokens.")

	print("Streaming response:\n", end="", flush=True)

	tokens = 0

	for chunk in service.stream(prompt,
			model="llama-3.3-70b-versatile",
			system="You are a precise, detailed technical writer.",
			):
		if not chunk:
			continue

		print(chunk, end="", flush=True)
		tokens += len(chunk.split())

	print("\n")

	health = service.health()
	print(f"\nPool health: {health.status}, active keys: {health.active_keys}")
	print(f"Approx tokens streamed: {tokens}")

	service.flush_tracking()

	if service._config.paths.base_dir:
		print(f"Data saved to {service._config.paths.base_dir}/")


if __name__ == "__main__":
	try:
		main()
	except GroqServiceError as e:
		print(f"\nError: {e}", file=sys.stderr)
		sys.exit(1)
