"""Basic one-shot chat with model selection and system prompt.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
    POOLGATE_DATA_DIR=./poolgate_data   # enables auto-persistence + file logs
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from exceptions.base import GroqServiceError
from services.provider_service import GroqService


load_dotenv()
os.environ.setdefault("POOLGATE_DATA_DIR", "./poolgate_data")


def main() -> None:
	service = GroqService()

	response = service.invoke(
		"What is the capital of France?",
		model="llama-3.3-70b-versatile",
		system="You are a helpful assistant. Answer concisely.",
		)
	print(f"Answer:  {response.text}")
	print(f"Tokens:  {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
	print(f"Latency: {response.latency:.3f}s  Model: {response.model}")

	service.flush_tracking()
	if service._config.data_dir:
		print(f"\nData saved to {service._config.data_dir}/")


if __name__ == "__main__":
	try:
		main()
	except GroqServiceError as e:
		print(f"Error: {e}")
