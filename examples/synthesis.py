"""Speech synthesis — convert text to audio with Orpheus TTS.

The synthesized audio is saved to poolgate_data/audio/output.mp3 so you can
play it back with any audio player.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from exceptions import GroqServiceError
from services.provider_service import GroqService

load_dotenv()

SYNTHESIS_MODEL = "canopylabs/orpheus-v1-english"
VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]


def main() -> None:
    text = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Hello! This is a test of PoolGate's text-to-speech synthesis."
    )
    voice = sys.argv[2] if len(sys.argv) > 2 else "tara"

    if voice not in VOICES:
        print(f"Unknown voice {voice!r}. Available: {', '.join(VOICES)}")
        sys.exit(1)

    service = GroqService()

    print(f"Synthesizing with voice={voice!r}...")
    result = service.synthesize(
        text=text,
        voice=voice,
        model=SYNTHESIS_MODEL,
        response_format="mp3",
    )

    # Save audio to poolgate_data/audio/ (or a local fallback)
    audio_dir: Path = service._config.paths.audio_dir or Path(".")
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_path = audio_dir / "output.mp3"

    with open(output_path, "wb") as f:
        f.write(result.audio)

    print(f"Audio saved: {output_path}  ({len(result.audio):,} bytes)")
    print(
        f"Voice: {result.voice}  Model: {result.model}  Latency: {result.latency:.3f}s"
    )

    service.flush_tracking()
    if service._config.paths.base_dir:
        print(f"\nTracking saved to {service._config.paths.base_dir}/tracking/")


if __name__ == "__main__":
    try:
        main()
    except GroqServiceError as e:
        print(f"Error: {e}")
        sys.exit(1)
