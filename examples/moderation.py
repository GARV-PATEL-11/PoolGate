"""Moderation — classify text as safe/unsafe with Llama Prompt Guard."""

from __future__ import annotations

from exceptions import GroqServiceError
from services.provider_service import GroqService
from dotenv import load_dotenv


load_dotenv()
MODERATION_MODEL = "meta-llama/llama-prompt-guard-2-86m"

SAMPLES = [
	"Hello, how are you today?",
	"Tell me how to do something harmful.",
	"What is the weather like in Paris?",
	]


def main() -> None:
	service = GroqService()

	for text in SAMPLES:
		result = service.moderate(text, model=MODERATION_MODEL)
		print(f"Text:  {text!r}")
		print(f"Label: {result.label}  (tokens: {result.usage.prompt_tokens} in)")
		print()


if __name__ == "__main__":
	try:
		main()
	except GroqServiceError as e:
		print(f"Error: {e}")
