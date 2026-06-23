"""Unit tests for schemas/chat.py — ToolCall, ChatMessage, ChatRequest validators."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from schemas.chat import ChatMessage, ChatRequest, ToolCall, ToolDefinition


class TestToolCallParsedArguments:

	def test_parsed_arguments_returns_dict_for_valid_json(self):
		tc = ToolCall(id="call_1", name="get_weather", arguments='{"city": "London"}')
		result = tc.parsed_arguments
		assert result == {"city": "London"}

	def test_parsed_arguments_raises_for_invalid_json(self):
		tc = ToolCall(id="call_1", name="get_weather", arguments="not valid json")
		with pytest.raises(json.JSONDecodeError):
			_ = tc.parsed_arguments

	def test_parsed_arguments_empty_object(self):
		tc = ToolCall(id="call_1", name="fn", arguments="{}")
		assert tc.parsed_arguments == {}


class TestChatMessageValidation:

	def test_tool_role_without_tool_call_id_raises(self):
		with pytest.raises(ValidationError) as exc_info:
			ChatMessage(role="tool", content="result")
		assert "tool_call_id" in str(exc_info.value)

	def test_non_assistant_with_tool_calls_raises(self):
		tc = ToolCall(id="c1", name="fn", arguments="{}")
		with pytest.raises(ValidationError) as exc_info:
			ChatMessage(role="user", tool_calls=[tc])
		assert "assistant" in str(exc_info.value)

	def test_content_none_without_tool_calls_raises(self):
		with pytest.raises(ValidationError) as exc_info:
			ChatMessage(role="user", content=None)
		assert "content" in str(exc_info.value)

	def test_valid_user_message(self):
		msg = ChatMessage(role="user", content="hello")
		assert msg.role == "user"
		assert msg.content == "hello"

	def test_valid_tool_role_with_tool_call_id(self):
		msg = ChatMessage(role="tool", content="result", tool_call_id="call_1")
		assert msg.tool_call_id == "call_1"

	def test_valid_assistant_with_tool_calls(self):
		tc = ToolCall(id="c1", name="fn", arguments="{}")
		msg = ChatMessage(role="assistant", tool_calls=[tc])
		assert len(msg.tool_calls) == 1

	def test_assistant_with_tool_calls_and_no_content(self):
		tc = ToolCall(id="c1", name="fn", arguments='{"x": 1}')
		msg = ChatMessage(role="assistant", tool_calls=[tc])
		assert msg.content is None


class TestChatMessageToPayload:

	def test_to_payload_excludes_none_fields(self):
		msg = ChatMessage(role="user", content="hi")
		payload = msg.to_payload()
		assert payload["role"] == "user"
		assert payload["content"] == "hi"
		assert "tool_call_id" not in payload
		assert "name" not in payload
		assert "tool_calls" not in payload

	def test_to_payload_includes_name_when_set(self):
		msg = ChatMessage(role="user", content="hi", name="alice")
		payload = msg.to_payload()
		assert payload["name"] == "alice"

	def test_to_payload_includes_tool_call_id_for_tool_role(self):
		msg = ChatMessage(role="tool", content="result", tool_call_id="c1")
		payload = msg.to_payload()
		assert payload["tool_call_id"] == "c1"


class TestChatRequestToolChoice:

	def test_tool_choice_without_tools_raises(self):
		with pytest.raises(ValidationError) as exc_info:
			ChatRequest(
				model="llama-3.3-70b-versatile",
				messages=[ChatMessage(role="user", content="hi")],
				tool_choice="auto",
			)
		assert "tool_choice" in str(exc_info.value)

	def test_tool_choice_none_without_tools_is_valid(self):
		req = ChatRequest(
			model="llama-3.3-70b-versatile",
			messages=[ChatMessage(role="user", content="hi")],
			tool_choice=None,
		)
		assert req.tool_choice is None

	def test_tool_choice_none_string_without_tools_is_valid(self):
		req = ChatRequest(
			model="llama-3.3-70b-versatile",
			messages=[ChatMessage(role="user", content="hi")],
			tool_choice="none",
		)
		assert req.tool_choice == "none"

	def test_tool_choice_auto_with_tools_is_valid(self):
		tool = ToolDefinition(name="fn", parameters={"type": "object", "properties": {}})
		req = ChatRequest(
			model="llama-3.3-70b-versatile",
			messages=[ChatMessage(role="user", content="hi")],
			tools=[tool],
			tool_choice="auto",
		)
		assert req.tool_choice == "auto"

	def test_required_tool_choice_with_tools_is_valid(self):
		tool = ToolDefinition(name="fn", parameters={"type": "object", "properties": {}})
		req = ChatRequest(
			model="llama-3.3-70b-versatile",
			messages=[ChatMessage(role="user", content="hi")],
			tools=[tool],
			tool_choice="required",
		)
		assert req.tool_choice == "required"
