"""Transcription and translation — speech-to-text with Whisper.

Usage:
    python examples/transcription.py audio.wav
    python examples/transcription.py audio.mp3 --translate

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from exceptions.base import GroqServiceError
from services.provider_service import GroqService


load_dotenv()


def main() -> None:
	if len(sys.argv) < 2:
		print("Usage: python examples/transcription.py <audio_file> [--translate]")
		print("  audio_file:  path to a .wav or .mp3 file")
		print("  --translate: translate to English instead of transcribing")
		sys.exit(0)

	audio_path = sys.argv[1]
	do_translate = "--translate" in sys.argv

	service = GroqService()

	with open(audio_path, "rb") as f:
		if do_translate:
			result = service.translate(f,
				model="whisper-large-v3",
				prompt="Translate the following audio to English.", )
			print(f"Translation: {result.text}")
		else:
			result = service.transcribe(f, model="whisper-large-v3", language="en", )
			print(f"Transcript: {result.text}")

	print(f"Model: {result.model}  Latency: {result.latency:.3f}s")

	service.flush_tracking()
	if service._config.paths.base_dir:
		print(f"\nData saved to {service._config.paths.base_dir}/")


if __name__ == "__main__":
	try:
		main()
	except FileNotFoundError as e:
		print(f"File not found: {e}")
		sys.exit(1)
	except GroqServiceError as e:
		print(f"Error: {e}")
		sys.exit(1)
