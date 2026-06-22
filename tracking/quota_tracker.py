"""
quota_tracker.py
-------------------
Tracks the provider's *own* authoritative quota numbers - e.g. Groq returns
`x-ratelimit-remaining-requests`, `x-ratelimit-remaining-tokens`, and reset
hints on every response header. This is intentionally separate from
request_tracker.py / token_tracker.py, which count *our* observed usage:
quota_tracker.py just remembers the last value the provider told us, per
model. The two numbers should usually agree; when they don't, the
provider's number should win for routing/key-rotation decisions, since it
accounts for things we can't see - other processes sharing the same key,
provider-side resets, etc.

Not persisted to disk on purpose: a quota snapshot is only meaningful at
the moment it was read - it goes stale the instant the next request is
sent, by anyone, on any process. There's nothing useful to keep in a daily
history file here.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field


@dataclass
class QuotaSnapshot:
	model: str
	remaining_rpd: int | None = None
	remaining_tpd: int | None = None
	reset_requests_seconds: float | None = None  # provider's own reset hint
	reset_tokens_seconds: float | None = None
	last_updated: float = field(default_factory=time.time)

	def to_dict(self) -> dict:
		return {
			"model": self.model,
			"remaining_rpd": self.remaining_rpd,
			"remaining_tpd": self.remaining_tpd,
			"reset_requests_seconds": self.reset_requests_seconds,
			"reset_tokens_seconds": self.reset_tokens_seconds,
			"last_updated": self.last_updated,
			}


class QuotaTracker:
	"""
	Thread-safe store of the most recently observed provider quota per
	model. Call `update_from_headers` right after every provider response.
	"""

	def __init__(self) -> None:
		self._quotas: dict[str, QuotaSnapshot] = {}
		self._lock = threading.Lock()

	def update(
			self,
			model: str,
			*,
			remaining_rpd: int | None = None,
			remaining_tpd: int | None = None,
			reset_requests_seconds: float | None = None,
			reset_tokens_seconds: float | None = None,
			) -> None:
		with self._lock:
			self._quotas[model] = QuotaSnapshot(
				model=model,
				remaining_rpd=remaining_rpd,
				remaining_tpd=remaining_tpd,
				reset_requests_seconds=reset_requests_seconds,
				reset_tokens_seconds=reset_tokens_seconds,
				)

	def update_from_headers(self, model: str, headers: dict) -> None:
		"""
		Convenience parser for Groq-style rate-limit headers:
		  x-ratelimit-remaining-requests
		  x-ratelimit-remaining-tokens
		  x-ratelimit-reset-requests   (e.g. "2m59.56s", or plain seconds)
		  x-ratelimit-reset-tokens
		Missing headers are simply ignored - call `update()` directly if a
		provider uses a different header scheme.
		"""

		def _int(key: str) -> int | None:
			v = headers.get(key)
			return int(v) if v is not None else None

		def _seconds(key: str) -> float | None:
			v = headers.get(key)
			if v is None:
				return None
			try:
				return float(v)
			except ValueError:
				return _parse_duration(v)

		self.update(
			model,
			remaining_rpd=_int("x-ratelimit-remaining-requests"),
			remaining_tpd=_int("x-ratelimit-remaining-tokens"),
			reset_requests_seconds=_seconds("x-ratelimit-reset-requests"),
			reset_tokens_seconds=_seconds("x-ratelimit-reset-tokens"),
			)

	def get(self, model: str) -> dict | None:
		with self._lock:
			snap = self._quotas.get(model)
			return snap.to_dict() if snap else None

	def check_quota(self, model: str) -> bool:
		return not self.is_exhausted(model)

	def consume(self, model: str, *, requests: int = 1, tokens: int = 0) -> None:
		with self._lock:
			snap = self._quotas.get(model)
			# No provider snapshot yet; optimistic local consumption is not meaningful.
			if snap is None:
				return
			if snap.remaining_rpd is not None:
				snap.remaining_rpd = max(0, snap.remaining_rpd - requests)
			if snap.remaining_tpd is not None:
				snap.remaining_tpd = max(0, snap.remaining_tpd - tokens)
			snap.last_updated = time.time()

	def get_remaining(self, model: str) -> dict:
		snap = self.get(model) or {}
		return {
			"remaining_rpd": snap.get("remaining_rpd"),
			"remaining_tpd": snap.get("remaining_tpd"),
			}

	def snapshot_all(self) -> list[dict]:
		with self._lock:
			return [s.to_dict() for s in self._quotas.values()]

	def is_exhausted(self, model: str) -> bool:
		"""True only when the provider has explicitly told us 0 remain."""
		with self._lock:
			snap = self._quotas.get(model)
		if not snap:
			return False
		return snap.remaining_rpd == 0 or snap.remaining_tpd == 0


def _parse_duration(value: str) -> float | None:
	"""Parses Groq-style durations like '2m59.56s' or '12.3s' into seconds."""
	match = re.match(r"^(?:(\d+)m)?(?:([\d.]+)s)?$", value.strip())
	if not match:
		return None
	minutes = float(match.group(1)) if match.group(1) else 0.0
	seconds = float(match.group(2)) if match.group(2) else 0.0
	return minutes * 60 + seconds
