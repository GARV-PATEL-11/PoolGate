"""Custom scheduling — switch key-selection strategy at runtime.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=2          # strategies are most visible with multiple keys
    GROQ_API_KEY_01=gsk_...
    GROQ_API_KEY_02=gsk_...
    POOLGATE_DATA_DIR=./poolgate_data
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from exceptions.base import GroqServiceError
from schedulers.scheduling_strategies import SchedulingStrategyType
from services.provider_service import GroqService


load_dotenv()
os.environ.setdefault("POOLGATE_DATA_DIR", "./poolgate_data")


def main() -> None:
	service = GroqService()

	# HEALTH_AWARE (default) — picks the key with the best composite score
	response = service.invoke("What is 1 + 1?", model="llama-3.3-70b-versatile")
	print(f"Health-aware result: {response.text.strip()}")

	# ROUND_ROBIN — equal traffic distribution across all keys
	service._scheduler.set_strategy(SchedulingStrategyType.ROUND_ROBIN)
	response = service.invoke("What is 2 + 2?", model="llama-3.3-70b-versatile")
	print(f"Round-robin result:  {response.text.strip()}")

	# LEAST_USED — minimize hotspots
	service._scheduler.set_strategy(SchedulingStrategyType.LEAST_USED)
	response = service.invoke("What is 3 + 3?", model="llama-3.3-70b-versatile")
	print(f"Least-used result:   {response.text.strip()}")

	# Restore default
	service._scheduler.set_strategy(SchedulingStrategyType.HEALTH_AWARE)

	print("\nKey pool status:")
	for key in service.get_key_pool_status():
		masked = key.get("masked_key", key.get("key_id", "?"))
		print(f"  {masked}: status={key.get('status')} rpm={key.get('requests_per_minute', 0)}")

	service.flush_tracking()
	if service._config.data_dir:
		print(f"\nData saved to {service._config.data_dir}/")


if __name__ == "__main__":
	try:
		main()
	except GroqServiceError as e:
		print(f"Error: {e}")
