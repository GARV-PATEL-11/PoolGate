"""Custom scheduling — switch key-selection strategy at runtime.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=2          # strategies are most visible with multiple keys
    GROQ_API_KEY_01=gsk_...
    GROQ_API_KEY_02=gsk_...
"""

from __future__ import annotations

from dotenv import load_dotenv

from exceptions.base import GroqServiceError
from schedulers.scheduling_strategies import SchedulingStrategyType
from services.provider_service import GroqService


load_dotenv()

PROMPTS = [
	"What is 2 + 2?",
	"Name the planets in our solar system.",
	"Write a haiku about Python.",
	"What is the boiling point of water in Celsius?",
	"Who wrote Romeo and Juliet?",
	"What is the capital of France?",
	"Explain photosynthesis in simple terms.",
	"What is the speed of light?",
	"Define gravity.",
	"Translate 'good morning' into Spanish.",
	"What is the square root of 144?",
	"List the continents of the world.",
	"Who discovered gravity?",
	"What is AI?",
	"Write a short joke about programmers.",
	"What is the largest ocean on Earth?",
	"Explain recursion in programming.",
	"What is HTTP?",
	"What is 10 factorial?",
	"Name three programming languages.",
	"What is a database?",
	"What is machine learning?",
	"Convert 100 Celsius to Fahrenheit.",
	"What is the tallest mountain in the world?",
	"Who is Albert Einstein?",
	"What is an API?",
	"Explain JSON in one sentence.",
	"What is 5 * 6?",
	"What is the meaning of life (philosophically)?",
	"Write a Python one-liner to reverse a string.",
	"What is Docker?",
	"What is Git used for?",
	"Explain cloud computing briefly.",
	"What is cybersecurity?",
	"Name 5 animals that live in the ocean.",
	"What is the Pythagorean theorem?",
]


def main() -> None:
	service = GroqService()

	# HEALTH_AWARE (default) — batch run
	print("\n=== HEALTH_AWARE (BATCH) ===")
	for i, prompt in enumerate(PROMPTS, 1):
		response = service.invoke(prompt, model="llama-3.3-70b-versatile")
		print(f"[{i}] {prompt}")
		print(f"    → {response.text.strip()}")

	# ROUND_ROBIN — equal traffic distribution across all keys
	service._scheduler.set_strategy(SchedulingStrategyType.ROUND_ROBIN)
	print("\n=== ROUND_ROBIN (BATCH) ===")
	for i, prompt in enumerate(PROMPTS, 1):
		response = service.invoke(prompt, model="llama-3.3-70b-versatile")
		print(f"[{i}] {prompt}")
		print(f"    → {response.text.strip()}")

	# LEAST_USED — minimize hotspots
	service._scheduler.set_strategy(SchedulingStrategyType.LEAST_USED)
	print("\n=== LEAST_USED (BATCH) ===")
	for i, prompt in enumerate(PROMPTS, 1):
		response = service.invoke(prompt, model="llama-3.3-70b-versatile")
		print(f"[{i}] {prompt}")
		print(f"    → {response.text.strip()}")

	# Restore default
	service._scheduler.set_strategy(SchedulingStrategyType.HEALTH_AWARE)

	print("\nKey pool status:")
	for key in service.get_key_pool_status():
		masked = key.get("masked_key", key.get("key_id", "?"))
		print(f"  {masked}: status={key.get('status')} rpm={key.get('requests_per_minute', 0)}")

	service.flush_tracking()
	if service._config.paths.base_dir:
		print(f"\nData saved to {service._config.paths.base_dir}/")


if __name__ == "__main__":
	try:
		main()
	except GroqServiceError as e:
		print(f"Error: {e}")
