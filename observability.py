"""
Structured logging and observability utilities.
Raw API keys are NEVER logged; they are always masked.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


_KEY_PATTERN = re.compile(r"(gsk_[A-Za-z0-9]{4})[A-Za-z0-9]+([A-Za-z0-9]{4})")


def mask_key(key: str) -> str:
	"""Return a safely masked version: gsk_****abcd."""
	m = _KEY_PATTERN.match(key)
	if m:
		return f"{m.group(1)}****{m.group(2)}"
	if len(key) > 8:
		return key[:4] + "****" + key[-4:]
	return "****"


_LOG_FMT = logging.Formatter(
	"%(asctime)s [%(levelname)s] %(name)s — %(message)s",
	datefmt="%Y-%m-%dT%H:%M:%S",
)

# 10 MB per file, keep 5 rotations
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5


class _RequestFilter(logging.Filter):
	"""Only passes records whose message starts with a request lifecycle keyword."""

	_PREFIXES = ("request_start", "request_end", "request_retry", "request_failure", "routing_decision")

	def filter(self, record: logging.LogRecord) -> bool:
		return any(record.getMessage().startswith(p) for p in self._PREFIXES)


def _file_handler(path: str, level: int, *, request_only: bool = False) -> logging.Handler:
	handler: logging.Handler = logging.handlers.RotatingFileHandler(
		path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8",
	)
	handler.setLevel(level)
	handler.setFormatter(_LOG_FMT)
	if request_only:
		handler.addFilter(_RequestFilter())
	return handler


def _make_logger(
		name: str,
		level: str = "INFO",
		debug_mode: bool = False,
		log_dir: str | None = None,
) -> logging.Logger:
	logger = logging.getLogger(name)
	if not logger.handlers:
		handler = logging.StreamHandler(sys.stdout)
		handler.setFormatter(_LOG_FMT)
		logger.addHandler(handler)

	if log_dir and not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers):
		os.makedirs(log_dir, exist_ok=True)
		logger.addHandler(_file_handler(os.path.join(log_dir, "general.log"), logging.DEBUG))
		logger.addHandler(_file_handler(os.path.join(log_dir, "info.log"), logging.INFO))
		logger.addHandler(_file_handler(os.path.join(log_dir, "error.log"), logging.ERROR))
		logger.addHandler(_file_handler(os.path.join(log_dir, "request.log"), logging.INFO, request_only=True))
		if debug_mode:
			logger.addHandler(_file_handler(os.path.join(log_dir, "debug.log"), logging.DEBUG))

	effective_level = logging.DEBUG if debug_mode else getattr(logging, level, logging.INFO)
	logger.setLevel(effective_level)
	return logger


@dataclass
class RequestContext:
	"""Carries per-request metadata for structured logging."""

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


class ObservabilityLogger:
	"""Thin wrapper around stdlib logging that injects request context."""

	def __init__(
			self,
			name: str = "groq_pool",
			level: str = "INFO",
			debug_mode: bool = False,
			log_dir: str | None = None,
			) -> None:
		self._log = _make_logger(name, level, debug_mode, log_dir=log_dir)
		self.debug_mode = debug_mode

	def _fmt(self, msg: str, ctx: RequestContext | None) -> str:
		if ctx:
			parts = " ".join(f"{k}={v}" for k, v in ctx.as_dict().items())
			return f"{msg} | {parts}"
		return msg

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
		start = time.perf_counter()
		self.debug(f"START {operation}", ctx)
		try:
			yield
		finally:
			elapsed = time.perf_counter() - start
			self.debug(f"END {operation} latency={elapsed:.3f}s", ctx)


# Module-level singleton — replaced by GroqService with proper config
_default_logger = ObservabilityLogger()


def get_logger(
		name: str = "groq_pool",
		level: str = "INFO",
		debug_mode: bool = False,
		log_dir: str | None = None,
		) -> ObservabilityLogger:
	return ObservabilityLogger(name, level, debug_mode, log_dir=log_dir)


class StructuredLogger(ObservabilityLogger):
	"""Spec-facing request lifecycle logger built on ObservabilityLogger."""

	def log_request_start(self, ctx: RequestContext) -> None:
		self.info("request_start", ctx)

	def log_request_end(
			self, ctx: RequestContext, *, latency: float, tokens_in: int = 0, tokens_out: int = 0,
			) -> None:
		ctx.extra.update({"latency": latency, "tokens_in": tokens_in, "tokens_out": tokens_out})
		self.info("request_end", ctx)

	def log_retry(self, ctx: RequestContext, *, attempt: int, error: BaseException | str) -> None:
		ctx.extra.update({"attempt": attempt, "error": str(error)})
		self.warning("request_retry", ctx)

	def log_failure(self, ctx: RequestContext, *, error: BaseException | str) -> None:
		ctx.extra.update({"error": str(error)})
		self.error("request_failure", ctx)

	def log_routing_decision(self, ctx: RequestContext, *, api_key_id: str, strategy: str) -> None:
		ctx.extra.update({"api_key_id": api_key_id, "strategy": strategy})
		self.debug("routing_decision", ctx)


@dataclass
class MetricsCollector:
	"""Small in-memory metrics sink suitable for tests and process-local snapshots."""

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


@dataclass
class TraceRecord:
	trace_id: str
	operation: str
	started_at: float = field(default_factory=time.perf_counter)
	ended_at: float | None = None
	metadata: dict[str, Any] = field(default_factory=dict)

	@property
	def duration(self) -> float | None:
		if self.ended_at is None:
			return None
		return self.ended_at - self.started_at


class Tracer:
	"""Minimal process-local tracer for service/request spans."""

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
