"""
tool_client.py — ToolClient

Function / tool calling.  Uses the same chat completions endpoint as ChatClient
but includes tools=[...] and tool_choice in the request body.

This client handles exactly one model invocation (request → GroqResponse).
Multi-turn tool loops — where the caller feeds tool_result messages back into
the next invocation — are orchestrated upstream.

Supported models:
  llama-3.1-8b-instant
  llama-3.3-70b-versatile
  meta-llama/llama-4-scout-17b-16e-instruct
  openai/gpt-oss-20b
  openai/gpt-oss-120b
  qwen/qwen3-32b
  qwen/qwen3.6-27b
  groq/compound
  groq/compound-mini

Public methods:
  invoke_tools()   — blocking tool-calling completion
  async_invoke_tools()  — async tool-calling completion
"""

from __future__ import annotations

import time
from abc import ABC
from typing import Any

from clients.base import (_choice_text, _first_choice, _new_rid, _parse_finish_reason, _parse_usage, BaseGroqClient)
from clients.capabilities import ToolCallingCapability
from schemas.runtime import GroqResponse, RequestConfig


class ToolClient(BaseGroqClient, ToolCallingCapability, ABC):
	"""
	Stateless client for function / tool calling.

	tools
		List of tool definitions following the OpenAI function-calling schema:
		[{"type": "function", "function": {"name": ..., "description": ...,
										   "parameters": {...}}}]

	tool_choice
		Controls which tool the model may invoke:
		  "auto"       — model decides (default)
		  "none"       — model must not call any tool
		  "required"   — model must call at least one tool
		  {"type": "function", "function": {"name": "<name>"}}
					   — model must call the named tool
	"""

	# ------------------------------------------------------------------
	# Sync
	# ------------------------------------------------------------------

	def invoke_tools(
			self,
			api_key: str,
			model: str,
			messages: list[dict[str, Any]],
			config: RequestConfig,
			session_id: str,
			api_key_id: str,
			tools: list[dict[str, Any]],
			tool_choice: str | dict[str, Any] = "auto",
			request_id: str | None = None,
			) -> GroqResponse:
		"""
		Blocking tool-calling completion.

		When finish_reason is TOOL_CALLS, inspect raw_response.choices[0]
		.message.tool_calls for the list of function calls the model wants
		to make.  Execute them, append the results as tool messages, then
		call invoke_tools() again to let the model continue.
		"""
		rid = _new_rid(request_id)
		client = self._sync_sdk(api_key)
		start = time.perf_counter()

		try:
			completion = client.chat.completions.create(
				model=model,
				messages=messages,  # type: ignore[arg-type]
				temperature=config.temperature,
				top_p=config.top_p,
				max_tokens=config.max_tokens,
				seed=config.seed,
				stop=config.stop,
				timeout=config.timeout,
				tools=tools,  # type: ignore[arg-type]
				tool_choice=tool_choice,  # type: ignore[arg-type]
				stream=False,
				)
		except Exception as exc:
			self._handle_sdk_error(exc, rid, api_key_id)

		latency = time.perf_counter() - start
		choice = _first_choice(completion, rid)
		return GroqResponse(
			text=_choice_text(choice, rid),
			model=model,
			usage=_parse_usage(completion),
			latency=latency,
			session_id=session_id,
			request_id=rid,
			api_key_id=api_key_id,
			finish_reason=_parse_finish_reason(getattr(choice, "finish_reason", None)),
			raw_response=completion,
			)

	# ------------------------------------------------------------------
	# Async
	# ------------------------------------------------------------------

	async def async_invoke_tools(
			self,
			api_key: str,
			model: str,
			messages: list[dict[str, Any]],
			config: RequestConfig,
			session_id: str,
			api_key_id: str,
			tools: list[dict[str, Any]],
			tool_choice: str | dict[str, Any] = "auto",
			request_id: str | None = None,
			) -> GroqResponse:
		"""
		Async tool-calling completion.

		Identical contract to invoke_tools() — uses the native async Groq SDK.
		"""
		rid = _new_rid(request_id)
		client = self._async_sdk(api_key)
		start = time.perf_counter()

		try:
			completion = await client.chat.completions.create(
				model=model,
				messages=messages,  # type: ignore[arg-type]
				temperature=config.temperature,
				top_p=config.top_p,
				max_tokens=config.max_tokens,
				seed=config.seed,
				stop=config.stop,
				timeout=config.timeout,
				tools=tools,  # type: ignore[arg-type]
				tool_choice=tool_choice,  # type: ignore[arg-type]
				stream=False,
				)
		except Exception as exc:
			self._handle_sdk_error(exc, rid, api_key_id)

		latency = time.perf_counter() - start
		choice = _first_choice(completion, rid)
		return GroqResponse(
			text=_choice_text(choice, rid),
			model=model,
			usage=_parse_usage(completion),
			latency=latency,
			session_id=session_id,
			request_id=rid,
			api_key_id=api_key_id,
			finish_reason=_parse_finish_reason(getattr(choice, "finish_reason", None)),
			raw_response=completion,
			)
