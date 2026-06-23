"""Sessions — multi-turn conversation with a persistent session ID.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
"""

from __future__ import annotations

from dotenv import load_dotenv

from exceptions import GroqServiceError
from services.provider_service import GroqService

load_dotenv()

SESSION_ID = "demo-conversation"


def main() -> None:
    service = GroqService()

    turns = [
        {"role": "user", "content": "My name is Alice. Remember it."},
        {"role": "user", "content": "What is my name?"},
        {"role": "user", "content": "Give me a fun fact about the name Alice."},
    ]

    history: list[dict] = []
    for turn in turns:
        history.append(turn)
        response = service.chat(
            messages=history,
            model="llama-3.3-70b-versatile",
            session_id=SESSION_ID,
        )
        history.append({"role": "assistant", "content": response.text})
        print(f"User:      {turn['content']}")
        print(f"Assistant: {response.text[:120]}")
        print()

    stats = service.get_session_stats(SESSION_ID)
    if stats:
        total_tokens = stats.get("input_tokens", 0) + stats.get("output_tokens", 0)
        print(
            f"Session stats: {stats.get('total_requests')} requests, " f"{total_tokens} total tokens",
        )

    service.flush_tracking()
    if service._config.paths.base_dir:
        print(f"\nData saved to {service._config.paths.base_dir}/")


if __name__ == "__main__":
    try:
        main()
    except GroqServiceError as e:
        print(f"Error: {e}")
