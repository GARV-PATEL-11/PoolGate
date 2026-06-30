"""Unit tests for schemas/envelope.py — PoolGateRequest/Response validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from poolgate.schemas.requests.chat import ChatMessage, ChatRequest
from poolgate.schemas.responses.chat import ChatResponse
from poolgate.schemas.common import FinishReason
from poolgate.schemas.common.context import RequestContext
from poolgate.schemas.common.envelope import PoolGateRequest, PoolGateResponse
from poolgate.schemas.common.ops import ErrorResponse
from poolgate.schemas.responses.usage import TokenUsage


def _chat_request() -> ChatRequest:
    return ChatRequest(
        model="llama-3.3-70b-versatile",
        messages=[ChatMessage(role="user", content="hello")],
    )


def _request_context() -> RequestContext:
    return RequestContext(request_id="req-1", session_id="sess-1", api_key_id="key_0")


def _chat_response() -> ChatResponse:
    return ChatResponse(
        id="req-1",
        model="llama-3.3-70b-versatile",
        message=ChatMessage(role="assistant", content="hi"),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        latency_ms=120.0,
    )


class TestPoolGateRequestCapability:

    def test_capability_property_returns_request_type(self):
        req = PoolGateRequest(payload=_chat_request())
        assert req.capability == "chat"

    def test_capability_matches_payload_request_type(self):
        req = PoolGateRequest(payload=_chat_request())
        assert req.capability == req.payload.request_type


class TestPoolGateResponseValidation:

    def test_success_true_with_payload_is_valid(self):
        resp = PoolGateResponse(
            context=_request_context(),
            success=True,
            payload=_chat_response(),
            total_latency_ms=150.0,
        )
        assert resp.success is True
        assert resp.payload is not None

    def test_success_false_with_error_is_valid(self):
        resp = PoolGateResponse(
            context=_request_context(),
            success=False,
            error=ErrorResponse(error_code="rate_limit", message="Too many requests"),
            total_latency_ms=50.0,
        )
        assert resp.success is False
        assert resp.error is not None

    def test_success_true_with_error_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            PoolGateResponse(
                context=_request_context(),
                success=True,
                payload=_chat_response(),
                error=ErrorResponse(error_code="err", message="oops"),
                total_latency_ms=50.0,
            )
        assert "error" in str(exc_info.value).lower()

    def test_success_false_with_payload_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            PoolGateResponse(
                context=_request_context(),
                success=False,
                payload=_chat_response(),
                error=ErrorResponse(error_code="err", message="oops"),
                total_latency_ms=50.0,
            )
        assert "payload" in str(exc_info.value).lower()

    def test_success_false_without_error_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            PoolGateResponse(
                context=_request_context(),
                success=False,
                total_latency_ms=50.0,
            )
        assert "error" in str(exc_info.value).lower()
