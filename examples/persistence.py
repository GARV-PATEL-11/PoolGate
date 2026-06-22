"""Persistence — flush and reload tracker state across restarts.

PoolGate keeps usage counters (requests, tokens, per-model breakdown) in
memory. Pass a PersistenceService at construction so stats survive restarts.
"""

from __future__ import annotations

import os
import tempfile

from exceptions import GroqServiceError
from services.persistence_service import PersistenceService
from services.provider_service import GroqService
from dotenv import load_dotenv


load_dotenv()


def main() -> None:
	# Use a temp file so this example is self-contained; in production use a fixed path
	stats_path = os.path.join(tempfile.gettempdir(), "poolgate_usage.json")
	persistence = PersistenceService.json(stats_path)

	# Pass persistence at construction — GroqService loads prior history automatically
	service = GroqService(persistence=persistence)

	print(f"Stats file: {stats_path}")

	# Make a couple of calls to generate tracking data
	response = service.invoke(
		"Say 'hello' in three languages.",
		model="llama-3.3-70b-versatile",
		)
	print(f"Response: {response.text[:80]}...")

	# Flush current in-memory stats to disk
	service.flush_tracking()
	print("Stats flushed.")

	# Reload from disk to verify the round-trip
	saved = persistence.load_all()
	print(f"Persisted days: {sorted(saved.keys())}")
	if saved:
		latest_day = sorted(saved)[-1]
		print(f"Latest day ({latest_day}): {saved[latest_day]}")


if __name__ == "__main__":
	try:
		main()
	except GroqServiceError as e:
		print(f"Error: {e}")
