"""Basic one-shot chat with model selection and system prompt."""

from __future__ import annotations

from exceptions.base import GroqServiceError
from services.provider_service import GroqService
from dotenv import load_dotenv


load_dotenv()


def main() -> None:
	service = GroqService()

	response = service.invoke(
		"What is the capital of France?",
		model="llama-3.3-70b-versatile",
		system="You are a helpful assistant. Answer concisely.",
		)
	print(f"Answer: {response.text}")
	print(f"Tokens: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
	print(f"Latency: {response.latency:.3f}s  Model: {response.model}")


if __name__ == "__main__":
	try:
		main()
	except GroqServiceError as e:
		print(f"Error: {e}")
