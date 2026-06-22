"""
exceptions/keys.py
──────────────────
Exceptions related to the API-key pool lifecycle.


NoAvailableAPIKeyError — kept fully compatible, default message preserved
APIKeyDisabledError    — kept compatible; key_id='unknown' bug documented
                         and fixed (see class docstring)

APIKeyError — base class so callers can catch all key-pool errors in one
              except clause without catching unrelated GroqServiceErrors
"""

from __future__ import annotations

from exceptions.base import GroqServiceError


class APIKeyError(GroqServiceError):
	"""
	Base class for all API-key lifecycle exceptions.

	Catch this to handle any key-pool error with a single except clause.

	# No override needed — GroqServiceError.__init__ is sufficient.
	"""


class NoAvailableAPIKeyError(APIKeyError):
	"""
	Raised when all API keys are rate-limited, cooling down, or failed.

	Raised in schedulers/request_scheduler.py:138 (_select_key) when
	_gather_candidates() returns an empty list.

	Attributes
	----------
	total_keys   : number of keys registered in the pool
	reason_counts: breakdown of why each key is unavailable,
							   e.g. {"rate_limited": 3, "cooling": 1, "disabled": 2}
	"""

	def __init__(
			self,
			message: str = "No healthy API key available",
			total_keys: int = 0,
			reason_counts: dict[str, int] | None = None,
			request_id: str | None = None,
			) -> None:
		self.total_keys = total_keys
		self.reason_counts = reason_counts or {}
		super().__init__(message, request_id)


class APIKeyDisabledError(APIKeyError):
	"""
	Raised when an API key is explicitly disabled (401 / 403).

	Attributes
	----------
	key_id      : identifier of the disabled key (was always 'unknown')
	status_code : HTTP status code returned by Groq (401 or 403)
	"""

	def __init__(
			self,
			key_id: str,
			status_code: int | None = None,
			request_id: str | None = None,
			) -> None:
		self.key_id = key_id
		self.status_code = status_code
		super().__init__(
			f"API key '{key_id}' is disabled or unauthorized",
			request_id,
			)
