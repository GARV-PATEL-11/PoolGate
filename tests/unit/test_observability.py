"""Unit tests for observability.py."""

from __future__ import annotations

import logging

from core.logger_manager import (mask_key, ObservabilityLogger, RequestContext)


class TestMaskKey:

	def test_standard_groq_key_is_masked(self):
		key = "gsk_" + "A" * 40 + "ZZZZ"
		result = mask_key(key)
		assert result.startswith("gsk_")
		assert "****" in result
		assert result.endswith("ZZZZ")
		assert len(result) < len(key)

	def test_short_key_returns_stars(self):
		result = mask_key("short")
		assert result == "****"

	def test_non_standard_long_key_masked(self):
		key = "notgroq12345678"
		result = mask_key(key)
		assert "****" in result
		assert result.startswith("notg")
		assert result.endswith("5678")

	def test_raw_key_not_present_in_output(self):
		raw = "gsk_abcdefghijklmnopqrstuvwxyz1234"
		masked = mask_key(raw)
		assert raw not in masked


class TestRequestContext:

	def test_as_dict_contains_required_keys(self):
		ctx = RequestContext(
			request_id="req-1",
			session_id="sess-1",
			model="llama-3.3-70b-versatile",
			)
		d = ctx.as_dict()
		assert d["request_id"] == "req-1"
		assert d["session_id"] == "sess-1"
		assert d["model"] == "llama-3.3-70b-versatile"
		assert d["api_key_id"] == ""
		assert d["retry_count"] == 0

	def test_extra_fields_are_merged_into_dict(self):
		ctx = RequestContext(
			request_id="r",
			session_id="s",
			model="m",
			extra={"foo": "bar"},
			)
		d = ctx.as_dict()
		assert d["foo"] == "bar"

	def test_retry_count_is_included(self):
		ctx = RequestContext(
			request_id="r",
			session_id="s",
			model="m",
			retry_count=3,
			)
		assert ctx.as_dict()["retry_count"] == 3


class TestObservabilityLogger:

	def test_instantiation_with_defaults(self):
		logger = ObservabilityLogger()
		assert logger.debug_mode is False

	def test_instantiation_with_debug_mode(self):
		logger = ObservabilityLogger(debug_mode=True)
		assert logger.debug_mode is True

	def test_info_does_not_raise(self):
		logger = ObservabilityLogger()
		logger.info("test message")

	def test_warning_does_not_raise(self):
		logger = ObservabilityLogger()
		logger.warning("test warning")

	def test_error_does_not_raise(self):
		logger = ObservabilityLogger()
		logger.error("test error")

	def test_debug_only_logs_in_debug_mode(self, caplog):
		logger = ObservabilityLogger(name="test_debug", debug_mode=False)
		with caplog.at_level(logging.DEBUG, logger="test_debug"):
			logger.debug("should not appear")
		assert "should not appear" not in caplog.text

	def test_info_with_context_formats_message(self, caplog):
		logger = ObservabilityLogger(name="test_info_ctx")
		ctx = RequestContext(request_id="r1", session_id="s1", model="m1")
		with caplog.at_level(logging.INFO, logger="test_info_ctx"):
			logger.info("hello", ctx)
		assert "r1" in caplog.text
