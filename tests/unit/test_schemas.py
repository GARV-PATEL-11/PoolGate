"""Unit tests for schemas/ — Pydantic model validation and field constraints."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.ops import ErrorResponse, HealthStatus, RetryPolicy
from schemas.runtime import (
    BatchResult,
    BatchSummary,
    FinishReason,
    GroqResponse,
    RequestConfig,
    TokenUsage,
)


class TestHealthStatus:

    def test_healthy_status_is_valid(self):
        h = HealthStatus(status="healthy", uptime_seconds=10.0, active_keys=3, disabled_keys=0)
        assert h.status == "healthy"

    def test_degraded_status_is_valid(self):
        h = HealthStatus(status="degraded", uptime_seconds=5.0, active_keys=1, disabled_keys=1)
        assert h.status == "degraded"

    def test_unhealthy_status_is_valid(self):
        h = HealthStatus(status="unhealthy", uptime_seconds=0.1, active_keys=0, disabled_keys=2)
        assert h.status == "unhealthy"

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            HealthStatus(status="unknown", uptime_seconds=1.0, active_keys=0, disabled_keys=0)

    def test_negative_uptime_raises(self):
        with pytest.raises(ValidationError):
            HealthStatus(status="healthy", uptime_seconds=-1.0, active_keys=1, disabled_keys=0)


class TestRetryPolicy:

    def test_valid_policy_constructs(self):
        p = RetryPolicy(max_attempts=3, initial_backoff_seconds=1.0, max_backoff_seconds=10.0)
        assert p.max_attempts == 3

    def test_max_backoff_less_than_initial_raises(self):
        with pytest.raises(ValidationError):
            RetryPolicy(max_attempts=3, initial_backoff_seconds=5.0, max_backoff_seconds=2.0)

    def test_max_attempts_ge_one(self):
        with pytest.raises(ValidationError):
            RetryPolicy(max_attempts=0)


class TestErrorResponse:

    def test_basic_error_response(self):
        e = ErrorResponse(error_code="rate_limit_exceeded", message="Too many requests")
        assert e.error_code == "rate_limit_exceeded"

    def test_negative_retry_after_raises(self):
        with pytest.raises(ValidationError):
            ErrorResponse(error_code="rate_limit", message="x", retry_after=-1.0)

    def test_retry_after_zero_is_valid(self):
        e = ErrorResponse(error_code="rate_limit", message="x", retry_after=0.0)
        assert e.retry_after == 0.0


class TestTokenUsage:

    def test_default_construction_is_zero(self):
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_addition(self):
        u1 = TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        u2 = TokenUsage(prompt_tokens=2, completion_tokens=1, total_tokens=3)
        result = u1 + u2
        assert result.prompt_tokens == 7
        assert result.total_tokens == 11

    def test_negative_tokens_raise(self):
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-1)


class TestRequestConfig:

    def test_defaults_are_valid(self):
        cfg = RequestConfig()
        assert cfg.temperature == 1.0
        assert cfg.retries == 3

    def test_temperature_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            RequestConfig(temperature=3.0)

    def test_negative_retries_raises(self):
        with pytest.raises(ValidationError):
            RequestConfig(retries=-1)


class TestGroqResponse:

    def test_basic_construction(self):
        r = GroqResponse(
            text="hello",
            model="llama-3.3-70b-versatile",
            usage=TokenUsage(),
            latency=0.5,
            session_id="sess-1",
            api_key_id="key_0",
        )
        assert r.text == "hello"
        assert r.finish_reason == FinishReason.STOP


class TestBatchResult:

    def test_batch_result_construction(self):
        r = GroqResponse(
            text="ok",
            model="m",
            usage=TokenUsage(),
            latency=0.1,
            session_id="s",
            api_key_id="k",
        )
        br = BatchResult(index=0, prompt="hi", response=r, success=True)
        assert br.index == 0
        assert br.success is True


class TestBatchSummary:

    def test_batch_summary_aggregation(self):
        r = GroqResponse(
            text="ok",
            model="m",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            latency=0.2,
            session_id="s",
            api_key_id="k",
        )
        br = BatchResult(index=0, prompt="hi", response=r, success=True)
        total_usage = TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
        summary = BatchSummary(
            total=1,
            succeeded=1,
            failed=0,
            results=[br],
            total_tokens=total_usage,
            total_latency=0.2,
        )
        assert summary.total == 1
        assert summary.total_tokens.total_tokens == 7
