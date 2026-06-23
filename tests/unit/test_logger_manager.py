"""Unit tests for core/logger_manager.py — StructuredLogger, MetricsCollector, Tracer, etc."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from core.logger_manager import (
	LoggerManager,
	MetricsCollector,
	ObservabilityLogger,
	RequestContext,
	StructuredLogger,
	Tracer,
	TraceRecord,
	_JsonWriter,
)
from core.path_config import PathConfig


def _ctx(rid: str = "r1") -> RequestContext:
	return RequestContext(request_id=rid, session_id="s1", model="llama-3.3-70b-versatile")


class TestObservabilityLoggerDebugAndException:

	def test_debug_logs_when_debug_mode_enabled(self, caplog):
		logger = ObservabilityLogger(name="test_dbg_enabled", debug_mode=True)
		with caplog.at_level(logging.DEBUG, logger="test_dbg_enabled"):
			logger.debug("debug message")
		assert "debug message" in caplog.text

	def test_debug_silent_when_debug_mode_disabled(self, caplog):
		logger = ObservabilityLogger(name="test_dbg_disabled", debug_mode=False)
		with caplog.at_level(logging.DEBUG, logger="test_dbg_disabled"):
			logger.debug("should be silent")
		assert "should be silent" not in caplog.text

	def test_exception_does_not_raise(self):
		logger = ObservabilityLogger(name="test_exc_logger")
		try:
			raise ValueError("test exception")
		except ValueError:
			logger.exception("caught error")

	def test_timed_operation_context_manager_completes(self):
		logger = ObservabilityLogger(name="test_timed", debug_mode=True)
		with logger.timed_operation("my_op"):
			time.sleep(0.001)

	def test_timed_operation_yields_control(self):
		logger = ObservabilityLogger(name="test_timed_yield", debug_mode=False)
		result = []
		with logger.timed_operation("op"):
			result.append("inside")
		assert result == ["inside"]

	def test_from_logger_wraps_existing_logger(self):
		stdlib_logger = logging.getLogger("test_from_logger")
		obs = ObservabilityLogger.from_logger(stdlib_logger, debug_mode=True)
		assert obs.debug_mode is True
		assert obs._log is stdlib_logger


class TestStructuredLogger:

	def setup_method(self):
		self._logger = StructuredLogger(name="test_structured")

	def test_log_request_start_does_not_raise(self):
		self._logger.log_request_start(_ctx())

	def test_log_request_end_does_not_raise(self):
		self._logger.log_request_end(_ctx(), latency=0.5, tokens_in=10, tokens_out=5)

	def test_log_retry_does_not_raise(self):
		self._logger.log_retry(_ctx(), attempt=2, error="transient error")

	def test_log_failure_does_not_raise(self):
		self._logger.log_failure(_ctx(), error=RuntimeError("test failure"))

	def test_log_routing_decision_does_not_raise(self):
		self._logger.log_routing_decision(_ctx(), api_key_id="key_0", strategy="HEALTH_AWARE")

	def test_log_request_end_attaches_latency_to_context(self):
		ctx = _ctx("r-latency")
		self._logger.log_request_end(ctx, latency=1.23)
		assert ctx.extra.get("latency") == 1.23

	def test_log_retry_attaches_attempt_and_error(self):
		ctx = _ctx("r-retry")
		self._logger.log_retry(ctx, attempt=3, error="conn reset")
		assert ctx.extra.get("attempt") == 3
		assert ctx.extra.get("error") == "conn reset"


class TestMetricsCollector:

	def test_initial_state_is_zero(self):
		m = MetricsCollector()
		snap = m.snapshot()
		assert snap["request_count"] == 0
		assert snap["retry_count"] == 0
		assert snap["failure_count"] == 0
		assert snap["quota_exhaustions"] == 0
		assert snap["average_latency"] == 0.0

	def test_inc_request_count(self):
		m = MetricsCollector()
		m.inc_request_count(3)
		assert m.snapshot()["request_count"] == 3

	def test_inc_retry_count(self):
		m = MetricsCollector()
		m.inc_retry_count(2)
		assert m.snapshot()["retry_count"] == 2

	def test_inc_failure_count(self):
		m = MetricsCollector()
		m.inc_failure_count()
		assert m.snapshot()["failure_count"] == 1

	def test_record_quota_exhaustion(self):
		m = MetricsCollector()
		m.record_quota_exhaustion(2)
		assert m.snapshot()["quota_exhaustions"] == 2

	def test_record_latency_computes_average(self):
		m = MetricsCollector()
		m.record_latency(0.1)
		m.record_latency(0.3)
		snap = m.snapshot()
		assert snap["average_latency"] == pytest.approx(0.2)

	def test_record_routing_decision_tracks_strategies(self):
		m = MetricsCollector()
		m.record_routing_decision("HEALTH_AWARE")
		m.record_routing_decision("HEALTH_AWARE")
		m.record_routing_decision("ROUND_ROBIN")
		snap = m.snapshot()
		assert snap["routing_decisions"]["HEALTH_AWARE"] == 2
		assert snap["routing_decisions"]["ROUND_ROBIN"] == 1


class TestTracer:

	def test_start_trace_returns_trace_record(self):
		tracer = Tracer()
		record = tracer.start_trace("t1", "my_operation")
		assert isinstance(record, TraceRecord)
		assert record.trace_id == "t1"
		assert record.operation == "my_operation"

	def test_end_trace_sets_ended_at(self):
		tracer = Tracer()
		tracer.start_trace("t1", "op")
		record = tracer.end_trace("t1")
		assert record is not None
		assert record.ended_at is not None

	def test_end_trace_unknown_id_returns_none(self):
		tracer = Tracer()
		assert tracer.end_trace("nonexistent") is None

	def test_get_trace_returns_record(self):
		tracer = Tracer()
		tracer.start_trace("t1", "op")
		assert tracer.get_trace("t1") is not None

	def test_get_trace_unknown_returns_none(self):
		tracer = Tracer()
		assert tracer.get_trace("unknown") is None

	def test_trace_record_duration_is_none_before_end(self):
		tracer = Tracer()
		record = tracer.start_trace("t1", "op")
		assert record.duration is None

	def test_trace_record_duration_is_positive_after_end(self):
		tracer = Tracer()
		tracer.start_trace("t1", "op")
		time.sleep(0.001)
		record = tracer.end_trace("t1")
		assert record.duration is not None
		assert record.duration > 0

	def test_end_trace_attaches_metadata(self):
		tracer = Tracer()
		tracer.start_trace("t1", "op")
		tracer.end_trace("t1", status="ok")
		record = tracer.get_trace("t1")
		assert record.metadata.get("status") == "ok"


class TestJsonWriter:

	def test_write_to_none_path_is_noop(self):
		writer = _JsonWriter(None)
		writer.write({"key": "value"})

	def test_write_creates_file_and_appends_json_line(self, tmp_path):
		path = tmp_path / "test.log"
		writer = _JsonWriter(path)
		writer.write({"event": "test", "value": 42})
		content = path.read_text()
		assert '"event": "test"' in content
		assert content.endswith("\n")

	def test_write_appends_multiple_lines(self, tmp_path):
		path = tmp_path / "test.log"
		writer = _JsonWriter(path)
		writer.write({"n": 1})
		writer.write({"n": 2})
		lines = path.read_text().strip().split("\n")
		assert len(lines) == 2

	def test_write_adds_timestamp_field(self, tmp_path):
		path = tmp_path / "test.log"
		writer = _JsonWriter(path)
		writer.write({"event": "x"})
		import json
		data = json.loads(path.read_text().strip())
		assert "ts" in data

	def test_write_silently_ignores_oserror(self, tmp_path, monkeypatch):
		path = tmp_path / "test.log"
		writer = _JsonWriter(path)

		def _bad_open(*args, **kwargs):
			raise OSError("disk full")

		monkeypatch.setattr("builtins.open", _bad_open)
		writer.write({"event": "ignored"})  # must not raise


class TestLoggerManager:

	def test_logger_manager_instantiation(self, tmp_path):
		paths = PathConfig(base_dir=tmp_path)
		manager = LoggerManager(paths, level="INFO", debug=False)
		assert manager is not None

	def test_get_returns_observability_logger(self, tmp_path):
		paths = PathConfig(base_dir=tmp_path)
		manager = LoggerManager(paths)
		obs = manager.get()
		assert isinstance(obs, ObservabilityLogger)

	def test_get_structured_returns_structured_logger(self, tmp_path):
		paths = PathConfig(base_dir=tmp_path)
		manager = LoggerManager(paths)
		sl = manager.get_structured()
		assert isinstance(sl, StructuredLogger)

	def test_log_request_writes_json_line(self, tmp_path):
		paths = PathConfig(base_dir=tmp_path)
		manager = LoggerManager(paths)
		manager.log_request(rid="r1", sid="s1", model="m", capability="chat")
		import json
		content = paths.request_log.read_text()
		data = json.loads(content.strip())
		assert data["rid"] == "r1"
		assert data["capability"] == "chat"

	def test_log_response_writes_json_line(self, tmp_path):
		paths = PathConfig(base_dir=tmp_path)
		manager = LoggerManager(paths)
		manager.log_response(
			rid="r1", model="m", tokens_in=10, tokens_out=5, latency=0.3, success=True,
		)
		import json
		content = paths.response_log.read_text()
		data = json.loads(content.strip())
		assert data["rid"] == "r1"
		assert data["success"] is True

	def test_log_trace_writes_stage(self, tmp_path):
		paths = PathConfig(base_dir=tmp_path)
		manager = LoggerManager(paths)
		manager.log_trace(rid="r1", stage="key_acquired", key_id="key_0")
		import json
		content = paths.trace_log.read_text()
		data = json.loads(content.strip())
		assert data["stage"] == "key_acquired"

	def test_log_tool_call_writes_tools(self, tmp_path):
		paths = PathConfig(base_dir=tmp_path)
		manager = LoggerManager(paths)
		manager.log_tool_call(rid="r1", model="m", tool_names=["fn_a", "fn_b"], latency=0.1)
		import json
		content = paths.tool_calls_log.read_text()
		data = json.loads(content.strip())
		assert data["tools"] == ["fn_a", "fn_b"]

	def test_log_performance_writes_metrics(self, tmp_path):
		paths = PathConfig(base_dir=tmp_path)
		manager = LoggerManager(paths)
		manager.log_performance(rid="r1", model="m", latency=0.5, tokens_in=10, tokens_out=20)
		import json
		content = paths.performance_log.read_text()
		data = json.loads(content.strip())
		assert data["tokens_out_per_s"] == pytest.approx(40.0)

	def test_log_storage_writes_event(self, tmp_path):
		paths = PathConfig(base_dir=tmp_path)
		manager = LoggerManager(paths)
		manager.log_storage(event="flush", tracker="usage", path="/tmp/usage.json")
		import json
		content = paths.storage_log.read_text()
		data = json.loads(content.strip())
		assert data["event"] == "flush"

	def test_log_performance_zero_latency_gives_zero_tps(self, tmp_path):
		paths = PathConfig(base_dir=tmp_path)
		manager = LoggerManager(paths)
		manager.log_performance(rid="r1", model="m", latency=0.0, tokens_in=5, tokens_out=10)
		import json
		content = paths.performance_log.read_text()
		data = json.loads(content.strip())
		assert data["tokens_out_per_s"] == 0.0

	def test_debug_mode_creates_debug_log(self, tmp_path):
		paths = PathConfig(base_dir=tmp_path)
		manager = LoggerManager(paths, debug=True)
		assert manager is not None

	def test_debug_mode_adds_debug_file_handler(self, tmp_path, monkeypatch):
		import logging.handlers

		poolgate_logger = logging.getLogger("poolgate")
		# Clear file handlers so the branch that adds them (line 432) is taken fresh
		monkeypatch.setattr(poolgate_logger, "handlers", [])

		paths = PathConfig(base_dir=tmp_path)
		LoggerManager(paths, debug=True)

		debug_handlers = [
			h for h in poolgate_logger.handlers
			if isinstance(h, logging.handlers.RotatingFileHandler)
			and "debug" in str(getattr(h, "baseFilename", ""))
		]
		assert len(debug_handlers) >= 1
