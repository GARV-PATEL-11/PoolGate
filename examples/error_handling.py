"""Error handling — catch specific PoolGate exception types.

When POOLGATE_DATA_DIR is set, errors are also written to logs/error.log
alongside the normal application logs.

Environment (.env or shell):
    TOTAL_GROQ_KEYS=1
    GROQ_API_KEY_01=gsk_...
    POOLGATE_DATA_DIR=./poolgate_data
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from exceptions import (
	APIKeyDisabledError,
	GroqServiceError,
	InvalidRequestError,
	NoAvailableAPIKeyError,
	RateLimitExceededError,
	)
from services.provider_service import GroqService


load_dotenv()
os.environ.setdefault("POOLGATE_DATA_DIR", "./poolgate_data")


def safe_invoke(service: GroqService, prompt: str) -> str | None:
	try:
		response = service.invoke(prompt, model="llama-3.3-70b-versatile")
		return response.text

	except NoAvailableAPIKeyError as e:
		print(f"[NoAvailableAPIKeyError] Pool exhausted: {e.total_keys} keys")
		print(f"  Reason breakdown: {e.reason_counts}")
		return None

	except APIKeyDisabledError as e:
		print(f"[APIKeyDisabledError] Key {e.key_id!r} is disabled (HTTP {e.status_code})")
		return None

	except RateLimitExceededError as e:
		retry_msg = f", retry after {e.retry_after}s" if e.retry_after else ""
		print(f"[RateLimitExceededError] Rate limited{retry_msg}")
		return None

	except InvalidRequestError as e:
		print(f"[InvalidRequestError] Bad request: {e}")
		return None

	except GroqServiceError as e:
		print(f"[GroqServiceError] {type(e).__name__}: {e}")
		return None


def main() -> None:
	service = GroqService()

	result = safe_invoke(service, "Say hello in one word.")
	if result:
		print(f"Success: {result.strip()}")

	# Demonstrate invalid role — error is also written to logs/error.log
	try:
		service.chat(
			messages=[{"role": "narrator", "content": "Tell the story."}],
			model="llama-3.3-70b-versatile",
			)
	except InvalidRequestError as e:
		print(f"\nExpected error for invalid role: {type(e).__name__}: {e}")

	health = service.health()
	stats = service.get_global_stats()
	print(f"\nPool: {health.status}, active keys: {health.active_keys}")
	print(
		f"Requests: {stats['total_requests']} total, "
		f"{stats['successful_requests']} succeeded, "
		f"{stats['failed_requests']} failed",
		)

	service.flush_tracking()
	if service._config.log_dir:
		print(f"\nLogs written to {service._config.log_dir}/")
		print("  general.log — all messages")
		print("  error.log   — errors only")
		print("  request.log — request lifecycle events")


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"Unexpected error: {e}", file=sys.stderr)
		sys.exit(1)
