"""
core/path_config.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for all PoolGate filesystem paths.

``PathConfig`` is a frozen dataclass whose only root is ``base_dir``.
Every other path is derived from it via the ``/`` operator so there is
never any string joining.

Pass ``base_dir=None`` to disable all I/O (useful in tests and library use).
Pass ``base_dir=tmp_path`` in tests for full isolation from production paths.

Usage::

    paths = PathConfig()                               # default poolgate_data
    paths = PathConfig(base_dir=None)                  # all I/O disabled
    paths = PathConfig(base_dir=tmp_path)              # isolated test dir
    paths = PathConfig(base_dir=Path("/var/poolgate")) # custom root
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Hardcoded at import time from this file's location; no env-var lookup.
# __file__ = …/poolgate/core/path_config.py  →  ../ = project root
_DEFAULT_BASE_DIR: Path = Path(__file__).resolve().parent.parent / "poolgate_data"


@dataclass(frozen=True)
class PathConfig:
	"""Single source of truth for all PoolGate filesystem paths.

	Parameters
	----------
	base_dir:
		Root data directory.  Defaults to the hardcoded ``poolgate_data``
		path inside the project root.  Pass ``None`` to disable all I/O.
	log_dir_override:
		Optional explicit log directory.  Overrides ``<base_dir>/logs``.
	"""

	base_dir: Path | None = _DEFAULT_BASE_DIR
	log_dir_override: Path | None = None

	# ── flags ─────────────────────────────────────────────────────────────

	@property
	def persistence_enabled(self) -> bool:
		"""``True`` when a ``base_dir`` has been configured."""
		return self.base_dir is not None

	@property
	def logging_enabled(self) -> bool:
		"""``True`` when a log directory is resolvable."""
		return self.log_dir is not None

	# ── directory paths ───────────────────────────────────────────────────

	@property
	def log_dir(self) -> Path | None:
		"""Resolved log directory: override → ``<base_dir>/logs`` → None."""
		if self.log_dir_override is not None:
			return self.log_dir_override
		if self.base_dir is None:
			return None
		return self.base_dir / "logs"

	@property
	def tracking_dir(self) -> Path | None:
		"""Persistent state directory: ``<base_dir>/tracking``, or ``None``."""
		return self.base_dir / "tracking" if self.base_dir is not None else None

	@property
	def requests_dir(self) -> Path | None:
		"""Per-day JSONL request archive: ``<base_dir>/requests``, or ``None``."""
		return self.base_dir / "requests" if self.base_dir is not None else None

	@property
	def audio_dir(self) -> Path | None:
		"""Generated audio output: ``<base_dir>/audio``, or ``None``."""
		return self.base_dir / "audio" if self.base_dir is not None else None

	# ── log file paths ────────────────────────────────────────────────────

	@property
	def general_log(self) -> Path | None:
		return self.log_dir / "general.log" if self.log_dir is not None else None

	@property
	def error_log(self) -> Path | None:
		return self.log_dir / "error.log" if self.log_dir is not None else None

	@property
	def request_log(self) -> Path | None:
		return self.log_dir / "request.log" if self.log_dir is not None else None

	@property
	def response_log(self) -> Path | None:
		return self.log_dir / "response.log" if self.log_dir is not None else None

	@property
	def trace_log(self) -> Path | None:
		return self.log_dir / "trace.log" if self.log_dir is not None else None

	@property
	def tool_calls_log(self) -> Path | None:
		return self.log_dir / "tool_calls.log" if self.log_dir is not None else None

	@property
	def performance_log(self) -> Path | None:
		return self.log_dir / "performance.log" if self.log_dir is not None else None

	@property
	def storage_log(self) -> Path | None:
		return self.log_dir / "storage.log" if self.log_dir is not None else None

	@property
	def debug_log(self) -> Path | None:
		return self.log_dir / "debug.log" if self.log_dir is not None else None

	# ── tracking file paths ───────────────────────────────────────────────

	@property
	def usage_json(self) -> Path | None:
		return self.tracking_dir / "usage.json" if self.tracking_dir is not None else None

	@property
	def tokens_json(self) -> Path | None:
		return self.tracking_dir / "tokens.json" if self.tracking_dir is not None else None

	@property
	def account_json(self) -> Path | None:
		return self.tracking_dir / "account.json" if self.tracking_dir is not None else None

	# ── helpers ───────────────────────────────────────────────────────────

	def ensure_dirs(self) -> None:
		"""Create all configured directories (``exist_ok=True``). Safe to call repeatedly."""
		for d in (self.log_dir, self.tracking_dir, self.requests_dir, self.audio_dir):
			if d is not None:
				d.mkdir(parents=True, exist_ok=True)
