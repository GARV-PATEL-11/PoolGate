"""Moderation — classify text as safe/unsafe with Llama Prompt Guard.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
    POOLGATE_DATA_DIR=./poolgate_data
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from exceptions import GroqServiceError
from services.provider_service import GroqService

load_dotenv()
os.environ.setdefault("POOLGATE_DATA_DIR", "./poolgate_data")

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

    service.flush_tracking()
    if service._config.data_dir:
        print(f"Data saved to {service._config.data_dir}/")


if __name__ == "__main__":
    try:
        main()
    except GroqServiceError as e:
        print(f"Error: {e}")
