"""
core/path_config.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for all PoolGate filesystem paths.

``PathConfig`` is a plain frozen dataclass.  Every path is derived from the
two optional root fields supplied by the caller.  There is no environment-
variable resolution, no computed default, and no fallback logic of any kind.
Callers that need a default data directory must compute and pass it themselves.

Usage::

    # Fully disabled (no I/O) — safe default for tests and library use:
    paths = PathConfig()

    # Caller-supplied root:
    paths = PathConfig(data_dir="/var/poolgate")
    paths.ensure_dirs()

    if paths.logging_enabled:
        open(paths.general_log, "a")
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PathConfig:
	"""Single source of truth for all PoolGate filesystem paths.

	Both fields default to ``None``, which disables persistence and logging
	respectively.  Callers must opt in by supplying an explicit path.

	Parameters
	----------
	data_dir:
		Root data directory.  When set, all sub-directories are derived from
		it.  When ``None`` (the default), persistence and logging are disabled.
	log_dir_override:
		Optional explicit log directory.  When set it takes precedence over
		``<data_dir>/logs``.  Ignored when ``data_dir`` is ``None`` and this
		field is also ``None``.
	"""

	# ── base dir resolution ─────────────────────────────────────────────

	@property
	def base_dir(self) -> str:
		"""
		Project root = parent.parent of this file.
		Adjust this file location and everything follows automatically.
		"""
		return os.path.abspath(
			os.path.join(os.path.dirname(__file__), "..", ".."),
			)

	base_directory = os.path.abspath(
		os.path.join(os.path.dirname(__file__), "..", ".."),
		)

	@property
	def _data_root(self) -> str:
		"""
		Final resolved data directory.
		"""
		if self.data_dir:
			return self.data_dir

		# preferred default
		fallback = os.path.join(base_directory, "poolgate_data")

		# second fallback (system-wide safe path)
		return fallback or "/var/poolgate"

	data_dir: str | os.path.join(self.base_dir, "poolgate_data")
	log_dir_override: str | None = None

	# ── derived flags ─────────────────────────────────────────────────────

	@property
	def persistence_enabled(self) -> bool:
		"""``True`` when a ``data_dir`` has been configured."""
		return self.data_dir is not None

	@property
	def logging_enabled(self) -> bool:
		"""``True`` when a log directory is resolvable."""
		return self.log_dir is not None

	# ── directory paths ───────────────────────────────────────────────────

	@property
	def log_dir(self) -> str | None:
		"""Resolved log directory.

		Returns ``log_dir_override`` when set, ``<data_dir>/logs`` when
		``data_dir`` is set, otherwise ``None``.
		"""
		if self.log_dir_override:
			return self.log_dir_override
		if self.data_dir is None:
			return None
		return os.path.join(self.data_dir, "logs")

	@property
	def tracking_dir(self) -> str | None:
		"""Persistent state directory: ``<data_dir>/tracking``, or ``None``."""
		if self.data_dir is None:
			return None
		return os.path.join(self.data_dir, "tracking")

	@property
	def requests_dir(self) -> str | None:
		"""Per-day JSONL request archive directory: ``<data_dir>/requests``, or ``None``."""
		if self.data_dir is None:
			return None
		return os.path.join(self.data_dir, "requests")

	@property
	def audio_dir(self) -> str | None:
		"""Generated audio output directory: ``<data_dir>/audio``, or ``None``."""
		if self.data_dir is None:
			return None
		return os.path.join(self.data_dir, "audio")

	# ── log file paths ────────────────────────────────────────────────────
	#
	# Each property captures self.log_dir in a local variable before the None
	# guard so type checkers can narrow str | None → str at the os.path.join
	# call site (property re-access is not narrowed by type checkers).

	@property
	def general_log(self) -> str | None:
		"""General-purpose application log, or ``None`` when logging is disabled."""
		log_dir = self.log_dir
		if log_dir is None:
			return None
		return os.path.join(log_dir, "general.log")

	@property
	def error_log(self) -> str | None:
		"""Error-only log, or ``None`` when logging is disabled."""
		log_dir = self.log_dir
		if log_dir is None:
			return None
		return os.path.join(log_dir, "error.log")

	@property
	def request_log(self) -> str | None:
		"""Inbound request log, or ``None`` when logging is disabled."""
		log_dir = self.log_dir
		if log_dir is None:
			return None
		return os.path.join(log_dir, "request.log")

	@property
	def response_log(self) -> str | None:
		"""Outbound response log, or ``None`` when logging is disabled."""
		log_dir = self.log_dir
		if log_dir is None:
			return None
		return os.path.join(log_dir, "response.log")

	@property
	def trace_log(self) -> str | None:
		"""Execution trace log, or ``None`` when logging is disabled."""
		log_dir = self.log_dir
		if log_dir is None:
			return None
		return os.path.join(log_dir, "trace.log")

	@property
	def tool_calls_log(self) -> str | None:
		"""Tool / function call log, or ``None`` when logging is disabled."""
		log_dir = self.log_dir
		if log_dir is None:
			return None
		return os.path.join(log_dir, "tool_calls.log")

	@property
	def performance_log(self) -> str | None:
		"""Latency and throughput metrics log, or ``None`` when logging is disabled."""
		log_dir = self.log_dir
		if log_dir is None:
			return None
		return os.path.join(log_dir, "performance.log")

	@property
	def storage_log(self) -> str | None:
		"""Filesystem and persistence events log, or ``None`` when logging is disabled."""
		log_dir = self.log_dir
		if log_dir is None:
			return None
		return os.path.join(log_dir, "storage.log")

	# ── tracking file paths ───────────────────────────────────────────────

	@property
	def usage_json(self) -> str | None:
		"""Aggregate usage statistics, or ``None`` when persistence is disabled."""
		tracking_dir = self.tracking_dir
		if tracking_dir is None:
			return None
		return os.path.join(tracking_dir, "usage.json")

	@property
	def tokens_json(self) -> str | None:
		"""Token usage counters, or ``None`` when persistence is disabled."""
		tracking_dir = self.tracking_dir
		if tracking_dir is None:
			return None
		return os.path.join(tracking_dir, "tokens.json")

	@property
	def account_json(self) -> str | None:
		"""Account-level metadata, or ``None`` when persistence is disabled."""
		tracking_dir = self.tracking_dir
		if tracking_dir is None:
			return None
		return os.path.join(tracking_dir, "account.json")

	# ── helpers ───────────────────────────────────────────────────────────

	def ensure_dirs(self) -> None:
		"""Create all configured data directories (``exist_ok=True``).

		Safe to call multiple times.  Silently skips any directory that
		resolves to ``None``.
		"""
		for d in (self.log_dir, self.tracking_dir, self.requests_dir, self.audio_dir):
			if d is not None:
				os.makedirs(d, exist_ok=True)
