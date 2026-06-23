"""Operational health snapshots for PoolGate."""
from __future__ import annotations

import time
from typing import Any

from schemas.ops import HealthStatus


class HealthService:
	"""Build health responses from scheduler, session, and usage state."""

	def __init__(self, *, version: str | None = None) -> None:
		self._started_at = time.perf_counter()
		self._version = version

	def snapshot(
		self,
		*,
		key_status: list[dict[str, Any]],
		active_sessions: int = 0,
		global_stats: dict[str, Any] | None = None,
		details: dict[str, Any] | None = None,
		) -> HealthStatus:
		active_keys = sum(1 for key in key_status if key.get("status") == "active")
		disabled_keys = sum(1 for key in key_status if key.get("status") == "disabled")
		failed_keys = sum(1 for key in key_status if key.get("status") == "failed")
		total_keys = len(key_status)

		if total_keys == 0 or active_keys == 0:
			status = "unhealthy"
		elif disabled_keys or failed_keys or active_keys < total_keys:
			status = "degraded"
		else:
			status = "healthy"

		payload = {
			"keys": key_status,
			"active_sessions": active_sessions,
			"global_stats": global_stats or {},
			}
		if details:
			payload.update(details)

		return HealthStatus(
			status=status,
			version=self._version,
			uptime_seconds=time.perf_counter() - self._started_at,
			active_keys=active_keys,
			disabled_keys=disabled_keys,
			details=payload,
			)
