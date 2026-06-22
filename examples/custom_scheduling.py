"""Custom scheduling — switch key-selection strategy at runtime."""

from __future__ import annotations

from exceptions.base import GroqServiceError
from schedulers.scheduling_strategies import SchedulingStrategyType
from services.provider_service import GroqService
from dotenv import load_dotenv


load_dotenv()


def main() -> None:
	service = GroqService()

	# Default strategy is HEALTH_AWARE — picks the key with the best composite score
	response = service.invoke("What is 1 + 1?", model="llama-3.3-70b-versatile")
	print(f"Health-aware result: {response.text.strip()}")

	# Switch to round-robin for equal traffic distribution
	service._scheduler.set_strategy(SchedulingStrategyType.ROUND_ROBIN)
	response = service.invoke("What is 2 + 2?", model="llama-3.3-70b-versatile")
	print(f"Round-robin result: {response.text.strip()}")

	# Switch to least-used to minimize hotspots
	service._scheduler.set_strategy(SchedulingStrategyType.LEAST_USED)
	response = service.invoke("What is 3 + 3?", model="llama-3.3-70b-versatile")
	print(f"Least-used result: {response.text.strip()}")

	# Restore default
	service._scheduler.set_strategy(SchedulingStrategyType.HEALTH_AWARE)

	# Show key pool status after the calls
	pool = service.get_key_pool_status()
	for key in pool:
		masked = key.get("masked_key", "Active as It's Not In Masked")
		status = key.get("status", "Active as It's Not In Masked")
		rpm = key.get("requests_per_minute", 0)

		print(f"  Key {masked}: status={status} rpm={rpm}")


if __name__ == "__main__":
	try:
		main()
	except GroqServiceError as e:
		print(f"Error: {e}")
