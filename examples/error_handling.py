"""Error handling — catch specific PoolGate exception types."""

from __future__ import annotations

import sys

from exceptions import (
	APIKeyDisabledError,
	GroqServiceError,
	InvalidRequestError,
	NoAvailableAPIKeyError,
	RateLimitExceededError,
	)
from services.provider_service import GroqService
from dotenv import load_dotenv


load_dotenv()


def safe_invoke(service: GroqService, prompt: str) -> str | None:
	try:
		response = service.invoke(prompt, model="llama-3.3-70b-versatile")
		return response.text

	except NoAvailableAPIKeyError as e:
		# All keys are rate-limited, cooling down, or failed
		print(f"[NoAvailableAPIKeyError] Pool exhausted: {e.total_keys} keys")
		print(f"  Reason breakdown: {e.reason_counts}")
		return None

	except APIKeyDisabledError as e:
		# A specific key returned 401/403
		print(f"[APIKeyDisabledError] Key {e.key_id!r} is disabled (HTTP {e.status_code})")
		return None

	except RateLimitExceededError as e:
		# All retry attempts hit 429
		retry_msg = f", retry after {e.retry_after}s" if e.retry_after else ""
		print(f"[RateLimitExceededError] Rate limited{retry_msg}")
		return None

	except InvalidRequestError as e:
		# Bad input — e.g. invalid role, missing prompt
		print(f"[InvalidRequestError] Bad request: {e}")
		return None

	except GroqServiceError as e:
		# Catch-all for any other PoolGate error
		print(f"[GroqServiceError] {type(e).__name__}: {e}")
		return None


def main() -> None:
	service = GroqService()

	# Normal successful call
	result = safe_invoke(service, "Say hello in one word.")
	if result:
		print(f"Success: {result.strip()}")

	# Demonstrate invalid role error
	try:
		service.chat(
			messages=[{"role": "narrator", "content": "Tell the story."}],
			model="llama-3.3-70b-versatile",
			)
	except InvalidRequestError as e:
		print(f"\nExpected error for invalid role: {type(e).__name__}: {e}")

	# Show pool health after calls
	health = service.health()
	stats = service.get_global_stats()
	print(f"\nPool: {health.status}, active keys: {health.active_keys}")
	print(
		f"Requests: {stats['total_requests']} total, "
		f"{stats['successful_requests']} succeeded, "
		f"{stats['failed_requests']} failed",
		)


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		print(f"Unexpected error: {e}", file=sys.stderr)
		sys.exit(1)
