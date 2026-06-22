"""Sessions — multi-turn conversation with a persistent session ID."""

from __future__ import annotations

from exceptions import GroqServiceError
from services.provider_service import GroqService
from dotenv import load_dotenv


load_dotenv()
SESSION_ID = "demo-conversation"


def main() -> None:
	service = GroqService()

	# All calls with the same session_id are grouped for per-session accounting
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
		# Add assistant reply to history so the model sees it next turn
		history.append({"role": "assistant", "content": response.text})
		print(f"User:      {turn['content']}")
		print(f"Assistant: {response.text[:120]}")
		print()

	# Check accumulated session stats
	stats = service.get_session_stats(SESSION_ID)
	if stats:
		print(
			f"Session stats: {stats.get('total_requests')} requests, "
			f"{stats.get('input_tokens', 0) + stats.get('output_tokens', 0)} total tokens",
			)


if __name__ == "__main__":
	try:
		main()
	except GroqServiceError as e:
		print(f"Error: {e}")
