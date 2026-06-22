"""Unit tests for module-level helpers in clients/base.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from clients.base import (
    BaseGroqClient,
    _choice_text,
    _chunk_delta_text,
    _first_choice,
    _new_rid,
    _parse_chunk_usage,
    _parse_finish_reason,
    _parse_usage,
)
from exceptions.keys import APIKeyDisabledError
from exceptions.rate_limit import RateLimitExceededError
from exceptions.response import InvalidResponseError
from schemas.runtime import FinishReason, TokenUsage


class TestParseFinishReason:
    def test_stop(self):
        assert _parse_finish_reason("stop") == FinishReason.STOP

    def test_length(self):
        assert _parse_finish_reason("length") == FinishReason.LENGTH

    def test_tool_calls(self):
        assert _parse_finish_reason("tool_calls") == FinishReason.TOOL_CALLS

    def test_content_filter(self):
        assert _parse_finish_reason("content_filter") == FinishReason.CONTENT_FILTER

    def test_unknown_string_returns_unknown(self):
        assert _parse_finish_reason("weird_value") == FinishReason.UNKNOWN

    def test_none_returns_unknown(self):
        assert _parse_finish_reason(None) == FinishReason.UNKNOWN

    def test_empty_string_returns_unknown(self):
        assert _parse_finish_reason("") == FinishReason.UNKNOWN


class TestParseUsage:
    def test_normal_completion_returns_token_usage(self):
        completion = MagicMock()
        completion.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        usage = _parse_usage(completion)
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

    def test_none_usage_returns_empty_token_usage(self):
        completion = MagicMock()
        completion.usage = None
        usage = _parse_usage(completion)
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestParseChunkUsage:
    def test_chunk_without_x_groq_returns_none(self):
        chunk = MagicMock()
        chunk.x_groq = None
        assert _parse_chunk_usage(chunk) is None

    def test_chunk_with_x_groq_but_no_usage_returns_none(self):
        chunk = MagicMock()
        chunk.x_groq = MagicMock()
        chunk.x_groq.usage = None
        assert _parse_chunk_usage(chunk) is None

    def test_final_chunk_with_usage_returns_token_usage(self):
        chunk = MagicMock()
        chunk.x_groq = MagicMock()
        chunk.x_groq.usage = MagicMock(prompt_tokens=8, completion_tokens=12, total_tokens=20)
        usage = _parse_chunk_usage(chunk)
        assert usage is not None
        assert usage.prompt_tokens == 8
        assert usage.completion_tokens == 12

    def test_chunk_without_x_groq_attribute_returns_none(self):
        chunk = MagicMock(spec=[])  # no x_groq attribute
        assert _parse_chunk_usage(chunk) is None


class TestNewRid:
    def test_none_generates_non_empty_string(self):
        rid = _new_rid(None)
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_provided_id_returned_unchanged(self):
        assert _new_rid("my-request-id") == "my-request-id"

    def test_two_generated_ids_are_distinct(self):
        assert _new_rid(None) != _new_rid(None)


class TestFirstChoice:
    def test_returns_first_choice(self):
        choice = MagicMock()
        completion = MagicMock()
        completion.choices = [choice]
        result = _first_choice(completion, "rid")
        assert result is choice

    def test_empty_choices_raises_invalid_response_error(self):
        completion = MagicMock()
        completion.choices = []
        with pytest.raises(InvalidResponseError):
            _first_choice(completion, "rid")

    def test_none_choices_raises_invalid_response_error(self):
        completion = MagicMock()
        completion.choices = None
        with pytest.raises(InvalidResponseError):
            _first_choice(completion, "rid")


class TestChoiceText:
    def test_returns_message_content(self):
        choice = MagicMock()
        choice.message.content = "hello world"
        assert _choice_text(choice, "rid") == "hello world"

    def test_none_content_returns_empty_string(self):
        choice = MagicMock()
        choice.message.content = None
        assert _choice_text(choice, "rid") == ""

    def test_missing_message_raises_invalid_response_error(self):
        choice = MagicMock(spec=[])  # no .message attribute
        with pytest.raises(InvalidResponseError):
            _choice_text(choice, "rid")


class TestChunkDeltaText:
    def test_returns_delta_content(self):
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="chunk text"))]
        assert _chunk_delta_text(chunk, "rid") == "chunk text"

    def test_none_content_returns_empty_string(self):
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content=None))]
        assert _chunk_delta_text(chunk, "rid") == ""

    def test_empty_choices_raises_invalid_response_error(self):
        chunk = MagicMock()
        chunk.choices = []
        with pytest.raises(InvalidResponseError):
            _chunk_delta_text(chunk, "rid")


def _make_exc_with_status(status_code: int) -> Exception:
    """Create a real exception instance with a status_code attribute."""

    class FakeSDKError(Exception):
        pass

    err = FakeSDKError("sdk error")
    err.status_code = status_code  # type: ignore[attr-defined]
    return err


class TestHandleSdkError:
    def _make_client(self) -> BaseGroqClient:
        return BaseGroqClient()

    def test_auth_error_raises_api_key_disabled(self):
        client = self._make_client()
        exc = _make_exc_with_status(401)
        with pytest.raises(APIKeyDisabledError):
            client._handle_sdk_error(exc, "rid", "key_x")

    def test_403_also_raises_api_key_disabled(self):
        client = self._make_client()
        exc = _make_exc_with_status(403)
        with pytest.raises(APIKeyDisabledError):
            client._handle_sdk_error(exc, "rid", "key_x")

    def test_rate_limit_error_raises_rate_limit_exceeded(self):
        client = self._make_client()

        class RateLimitError(Exception):
            pass

        exc = RateLimitError("rate limited")
        exc.response = None
        with pytest.raises(RateLimitExceededError):
            client._handle_sdk_error(exc, "rid", "key_x")

    def test_other_exception_reraises(self):
        client = self._make_client()
        exc = ValueError("unexpected error")
        with pytest.raises(ValueError):
            client._handle_sdk_error(exc, "rid", "key_x")
