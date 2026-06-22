"""
logger_manager.py
─────────────────────────────────────────────────────────────────────────────
Dependency chain (strict, no exceptions)::

    PathConfig  →  LoggerManager  →  logging output

Nothing in this module reads environment variables or resolves filesystem
paths outside what ``PathConfig`` provides.

Public surface
~~~~~~~~~~~~~~

Data classes / value types
    ``RequestContext``   — per-request metadata injected into log lines
    ``MetricsCollector`` — in-process counters and latency histogram
    ``TraceRecord``      — span record for a single named operation
    ``Tracer``           — lightweight process-local span store

Logger wrappers
    ``ObservabilityLogger``  — context-aware wrapper around a stdlib logger
    ``StructuredLogger``     — typed lifecycle event helpers built on the above

Primary factory
    ``LoggerManager``    — creates and owns all handlers; returns the wrappers
                           above; writes structured JSON lines to category files

Key safety
    ``mask_key(key)``    — masks raw API key strings; used throughout the system

Example::

    paths   = PathConfig(data_dir="/var/poolgate")
    manager = LoggerManager(paths, level="INFO", debug=False)
    logger  = manager.get()                         # ObservabilityLogger
    slogger = manager.get_structured()              # StructuredLogger

    ctx = RequestContext(request_id="r1", session_id="s1", model="llama3")
    slogger.log_request_start(ctx)

    manager.log_request(rid="r1", sid="s1", model="llama3", capability="chat")
    manager.log_performance(rid="r1", model="llama3", latency=0.3,
                            tokens_in=10, tokens_out=5)
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from core.path_config import PathConfig


# ── constants ─────────────────────────────────────────────────────────────────

_LOG_FMT = logging.Formatter(
	"%(asctime)s [%(levelname)s] %(name)s — %(message)s",
	datefmt="%Y-%m-%dT%H:%M:%S",
	)

_MAX_BYTES = 100 * 1024 * 1024  # 100 MB per file
_BACKUP_COUNT = 5

# ── key masking ───────────────────────────────────────────────────────────────

_KEY_PATTERN = re.compile(r"(gsk_[A-Za-z0-9]{4})[A-Za-z0-9]+([A-Za-z0-9]{4})")


def mask_key(key: str) -> str:
	"""Return a safely masked version of a raw API key: ``gsk_****abcd``.

	Raw API keys are *never* passed to any logger; callers must mask first.
	"""
	m = _KEY_PATTERN.match(key)
	if m:
		return f"{m.group(1)}****{m.group(2)}"
	if len(key) > 8:
		return key[:4] + "****" + key[-4:]
	return "****"


# ── per-request metadata ──────────────────────────────────────────────────────

@dataclass
class RequestContext:
	"""Carries per-request metadata injected into every structured log line."""

	request_id: str
	session_id: str
	model: str
	api_key_id: str = ""
	retry_count: int = 0
	extra: dict[str, Any] = field(default_factory=dict)

	def as_dict(self) -> dict[str, Any]:
		return {
			"request_id": self.request_id,
			"session_id": self.session_id,
			"model": self.model,
			"api_key_id": self.api_key_id,
			"retry_count": self.retry_count,
			**self.extra,
			}


# ── observable logger wrapper ─────────────────────────────────────────────────

class ObservabilityLogger:
	"""Context-aware wrapper around a stdlib ``logging.Logger``.

	Instances are always obtained via ``LoggerManager.get()`` so that all
	file handlers are created in one place.  The constructor sets up
	console-only output; ``from_logger`` wraps a pre-configured logger
	produced by ``LoggerManager``.
	"""

	def __init__(
			self,
			name: str = "poolgate",
			level: str = "INFO",
			debug_mode: bool = False,
			) -> None:
		logger = logging.getLogger(name)
		if not logger.handlers:
			ch = logging.StreamHandler(sys.stdout)
			ch.setFormatter(_LOG_FMT)
			logger.addHandler(ch)
		effective = logging.DEBUG if debug_mode else getattr(logging, level, logging.INFO)
		logger.setLevel(effective)
		self._log = logger
		self.debug_mode = debug_mode

	@classmethod
	def from_logger(
			cls,
			logger: logging.Logger,
			debug_mode: bool = False,
			) -> "ObservabilityLogger":
		"""Wrap an already-configured stdlib logger (used by ``LoggerManager``)."""
		obj = object.__new__(cls)
		obj._log = logger
		obj.debug_mode = debug_mode
		return obj

	# ── internal ──────────────────────────────────────────────────────────
	@staticmethod
	def _fmt(msg: str, ctx: RequestContext | None) -> str:
		if ctx:
			parts = " ".join(f"{k}={v}" for k, v in ctx.as_dict().items())
			return f"{msg} | {parts}"
		return msg

	# ── public log methods ─────────────────────────────────────────────────

	def info(self, msg: str, ctx: RequestContext | None = None, **kw: Any) -> None:
		self._log.info(self._fmt(msg, ctx), **kw)

	def debug(self, msg: str, ctx: RequestContext | None = None, **kw: Any) -> None:
		if self.debug_mode:
			self._log.debug(self._fmt(msg, ctx), **kw)

	def warning(self, msg: str, ctx: RequestContext | None = None, **kw: Any) -> None:
		self._log.warning(self._fmt(msg, ctx), **kw)

	def error(self, msg: str, ctx: RequestContext | None = None, **kw: Any) -> None:
		self._log.error(self._fmt(msg, ctx), **kw)

	def exception(self, msg: str, ctx: RequestContext | None = None) -> None:
		self._log.exception(self._fmt(msg, ctx))

	@contextmanager
	def timed_operation(
			self,
			operation: str,
			ctx: RequestContext | None = None,
			) -> Generator[None, None, None]:
		"""Context manager that debug-logs start/end with elapsed time."""
		start = time.perf_counter()
		self.debug(f"START {operation}", ctx)
		try:
			yield
		finally:
			elapsed = time.perf_counter() - start
			self.debug(f"END {operation} latency={elapsed:.3f}s", ctx)


# ── structured lifecycle logger ───────────────────────────────────────────────

class StructuredLogger(ObservabilityLogger):
	"""Typed request-lifecycle helpers built on ``ObservabilityLogger``.

	Obtained via ``LoggerManager.get_structured()``.
	"""

	def log_request_start(self, ctx: RequestContext) -> None:
		self.info("request_start", ctx)

	def log_request_end(
			self,
			ctx: RequestContext,
			*,
			latency: float,
			tokens_in: int = 0,
			tokens_out: int = 0,
			) -> None:
		ctx.extra.update({"latency": latency, "tokens_in": tokens_in, "tokens_out": tokens_out})
		self.info("request_end", ctx)

	def log_retry(
			self,
			ctx: RequestContext,
			*,
			attempt: int,
			error: BaseException | str,
			) -> None:
		ctx.extra.update({"attempt": attempt, "error": str(error)})
		self.warning("request_retry", ctx)

	def log_failure(self, ctx: RequestContext, *, error: BaseException | str) -> None:
		ctx.extra.update({"error": str(error)})
		self.error("request_failure", ctx)

	def log_routing_decision(
			self,
			ctx: RequestContext,
			*,
			api_key_id: str,
			strategy: str,
			) -> None:
		ctx.extra.update({"api_key_id": api_key_id, "strategy": strategy})
		self.debug("routing_decision", ctx)


# ── in-process metrics ────────────────────────────────────────────────────────

@dataclass
class MetricsCollector:
	"""In-memory metrics sink suitable for tests and process-local snapshots."""

	request_count: int = 0
	retry_count: int = 0
	failure_count: int = 0
	quota_exhaustions: int = 0
	latencies: list[float] = field(default_factory=list)
	routing_decisions: dict[str, int] = field(default_factory=dict)

	def inc_request_count(self, amount: int = 1) -> None:
		self.request_count += amount

	def inc_retry_count(self, amount: int = 1) -> None:
		self.retry_count += amount

	def inc_failure_count(self, amount: int = 1) -> None:
		self.failure_count += amount

	def record_latency(self, latency: float) -> None:
		self.latencies.append(latency)

	def record_quota_exhaustion(self, amount: int = 1) -> None:
		self.quota_exhaustions += amount

	def record_routing_decision(self, strategy: str) -> None:
		self.routing_decisions[strategy] = self.routing_decisions.get(strategy, 0) + 1

	def snapshot(self) -> dict[str, Any]:
		avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
		return {
			"request_count": self.request_count,
			"retry_count": self.retry_count,
			"failure_count": self.failure_count,
			"quota_exhaustions": self.quota_exhaustions,
			"average_latency": avg_latency,
			"routing_decisions": dict(self.routing_decisions),
			}


# ── distributed tracing primitives ────────────────────────────────────────────

@dataclass
class TraceRecord:
	"""Span record for a single named operation."""

	trace_id: str
	operation: str
	started_at: float = field(default_factory=time.perf_counter)
	ended_at: float | None = None
	metadata: dict[str, Any] = field(default_factory=dict)

	@property
	def duration(self) -> float | None:
		"""Elapsed seconds, or ``None`` if the span has not ended."""
		if self.ended_at is None:
			return None
		return self.ended_at - self.started_at


class Tracer:
	"""Minimal process-local tracer for service and request spans."""

	def __init__(self) -> None:
		self._traces: dict[str, TraceRecord] = {}

	def start_trace(self, trace_id: str, operation: str, **metadata: Any) -> TraceRecord:
		record = TraceRecord(trace_id=trace_id, operation=operation, metadata=metadata)
		self._traces[trace_id] = record
		return record

	def end_trace(self, trace_id: str, **metadata: Any) -> TraceRecord | None:
		record = self._traces.get(trace_id)
		if record is None:
			return None
		record.metadata.update(metadata)
		record.ended_at = time.perf_counter()
		return record

	def get_trace(self, trace_id: str) -> TraceRecord | None:
		return self._traces.get(trace_id)


# ── internal JSON line writer ─────────────────────────────────────────────────

class _JsonWriter:
	"""Thread-safe append-only JSON-line writer to a single file.

	When ``path`` is ``None`` all writes are silently dropped, making it safe
	to instantiate even when logging is disabled.
	"""

	def __init__(self, path: str | None) -> None:
		self._path = path
		self._lock = threading.Lock()

	def write(self, payload: dict[str, Any]) -> None:
		if self._path is None:
			return
		payload.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
		line = json.dumps(payload, default=str) + "\n"
		try:
			with self._lock:
				with open(self._path, "a", encoding="utf-8") as f:
					f.write(line)
		except OSError:
			pass


# ── primary factory ───────────────────────────────────────────────────────────

class LoggerManager:
	"""Central owner of all logging handlers and structured log writers.

	One instance is created per ``GroqService``.  It is the *only* place in
	the codebase that creates ``RotatingFileHandler`` objects or constructs
	filesystem paths; every path is sourced exclusively from the
	``PathConfig`` it receives.

	``get()``           → ``ObservabilityLogger``  (human-readable text)
	``get_structured()``→ ``StructuredLogger``     (typed lifecycle events)
	``log_*()``         → JSON lines in category-specific files

	Log categories
	~~~~~~~~~~~~~~
	request     — outbound API request start metadata    → request.log
	response    — API response outcome                   → response.log
	trace       — lifecycle stage events                 → trace.log
	tool_calls  — tool-calling invocations               → tool_calls.log
	performance — latency / throughput stats             → performance.log
	storage     — persistence layer events               → storage.log
	"""

	def __init__(
			self,
			paths: PathConfig,
			level: str = "INFO",
			debug: bool = False,
			) -> None:
		self._paths = paths
		self._debug = debug
		self._stdlib_level = logging.DEBUG if debug else getattr(logging, level, logging.INFO)

		# Create directories before any handler opens a file.
		paths.ensure_dirs()

		self._stdlib_logger = self._configure_stdlib_logger()
		self._writers: dict[str, _JsonWriter] = {
			"request": _JsonWriter(paths.request_log),
			"response": _JsonWriter(paths.response_log),
			"trace": _JsonWriter(paths.trace_log),
			"tool_calls": _JsonWriter(paths.tool_calls_log),
			"performance": _JsonWriter(paths.performance_log),
			"storage": _JsonWriter(paths.storage_log),
			}

	# ── stdlib logger setup ───────────────────────────────────────────────

	def _configure_stdlib_logger(self) -> logging.Logger:
		logger = logging.getLogger("poolgate")
		logger.setLevel(self._stdlib_level)

		# Console handler — added only if one is not already attached.
		has_console = any(
			(
					isinstance(h, logging.StreamHandler)
					and not isinstance(h, logging.FileHandler)
			)
				for h in logger.handlers
			)
		if not has_console:
			ch = logging.StreamHandler(sys.stdout)
			ch.setLevel(self._stdlib_level)
			ch.setFormatter(_LOG_FMT)
			logger.addHandler(ch)

		# File handlers — added only once; all paths come from PathConfig.
		has_file = any(
			isinstance(h, logging.handlers.RotatingFileHandler)
				for h in logger.handlers
			)
		if not has_file and self._paths.general_log:
			logger.addHandler(self._fh(self._paths.general_log, logging.DEBUG))
			if self._paths.error_log:
				logger.addHandler(self._fh(self._paths.error_log, logging.ERROR))
			if self._debug and self._paths.log_dir:
				logger.addHandler(
					self._fh(
						os.path.join(self._paths.log_dir, "debug.log"),
						logging.DEBUG,
						),
					)

		return logger

	@staticmethod
	def _fh(path: str, level: int) -> logging.Handler:
		h: logging.Handler = logging.handlers.RotatingFileHandler(
			path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8",
			)
		h.setLevel(level)
		h.setFormatter(_LOG_FMT)
		return h

	# ── logger accessors ──────────────────────────────────────────────────

	def get(self, name: str = "poolgate") -> ObservabilityLogger:
		"""Return an ``ObservabilityLogger`` backed by the managed stdlib logger."""
		return ObservabilityLogger.from_logger(self._stdlib_logger, debug_mode=self._debug)

	def get_structured(self, name: str = "poolgate") -> StructuredLogger:
		"""Return a ``StructuredLogger`` backed by the managed stdlib logger."""
		return StructuredLogger.from_logger(self._stdlib_logger, debug_mode=self._debug)

	# ── structured JSON log methods ───────────────────────────────────────

	def _write(self, category: str, payload: dict[str, Any]) -> None:
		writer = self._writers.get(category)
		if writer:
			writer.write(payload)

	def log_request(
			self,
			*,
			rid: str,
			sid: str,
			model: str,
			capability: str,
			attempt: int = 1,
			) -> None:
		"""Log an outbound API request start."""
		self._write("request", {
			"rid": rid, "sid": sid, "model": model,
			"capability": capability, "attempt": attempt,
			},
			)

	def log_response(
			self,
			*,
			rid: str,
			model: str,
			tokens_in: int,
			tokens_out: int,
			latency: float,
			success: bool,
			finish_reason: str = "",
			error: str = "",
			) -> None:
		"""Log an API response outcome."""
		self._write("response", {
			"rid": rid, "model": model,
			"tokens_in": tokens_in, "tokens_out": tokens_out,
			"latency_s": round(latency, 4), "success": success,
			"finish_reason": finish_reason, "error": error,
			},
			)

	def log_trace(self, *, rid: str, stage: str, **extra: Any) -> None:
		"""Log a lifecycle stage event (``key_acquired``, ``retry``, ``exhausted``, …)."""
		self._write("trace", {"rid": rid, "stage": stage, **extra})

	def log_tool_call(
			self,
			*,
			rid: str,
			model: str,
			tool_names: list[str],
			latency: float,
			finish_reason: str = "",
			) -> None:
		"""Log a tool-calling invocation."""
		self._write("tool_calls", {
			"rid": rid, "model": model, "tools": tool_names,
			"latency_s": round(latency, 4), "finish_reason": finish_reason,
			},
			)

	def log_performance(
			self,
			*,
			rid: str,
			model: str,
			latency: float,
			tokens_in: int,
			tokens_out: int,
			) -> None:
		"""Log latency and throughput metrics for a completed request."""
		tps = round(tokens_out / latency, 1) if latency > 0 else 0.0
		self._write("performance", {
			"rid": rid, "model": model,
			"latency_s": round(latency, 4),
			"tokens_in": tokens_in,
			"tokens_out": tokens_out,
			"tokens_out_per_s": tps,
			},
			)

	def log_storage(
			self,
			*,
			event: str,
			tracker: str = "",
			path: str = "",
			**extra: Any,
			) -> None:
		"""Log a persistence layer event (``load``, ``flush``, ``error``)."""
		self._write("storage", {"event": event, "tracker": tracker, "path": path, **extra})
