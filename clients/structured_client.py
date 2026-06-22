"""
structured_client.py — StructuredClient

JSON-mode / schema-constrained structured output generation.
Uses the same chat completions endpoint as ChatClient but forces response_format
to either {"type": "json_object"} (plain JSON mode) or a full JSON schema dict.

Supported models:
  llama-3.1-8b-instant
  llama-3.3-70b-versatile
  meta-llama/llama-4-scout-17b-16e-instruct
  openai/gpt-oss-20b
  openai/gpt-oss-120b
  qwen/qwen3-32b
  qwen/qwen3.6-27b

Note: groq/compound and groq/compound-mini intentionally excluded — compound
models do not reliably honour JSON-schema constraints at time of writing.

Public methods:
  invoke_structured()   — blocking structured completion
  async_invoke_structured()  — async structured completion
"""

from __future__ import annotations

import time
from typing import Any

from clients.base import (
	BaseGroqClient,
	_choice_text,
	_first_choice,
	_new_rid,
	_parse_finish_reason,
	_parse_usage,
	)
from clients.capabilities import StructuredGenerationCapability
from schemas.runtime import GroqResponse, RequestConfig


def _build_response_format(json_schema: dict[str, Any] | None) -> dict[str, Any]:
	"""
	Translate an optional JSON schema dict into a Groq response_format payload.

	json_schema=None   → {"type": "json_object"}            (plain JSON mode)
	json_schema={...}  → {"type": "json_schema", ...}       (schema-constrained)
	"""
	if json_schema is None:
		return {"type": "json_object"}
	return {
		"type": "json_schema",
		"json_schema": {
			"name": json_schema.get("title", "response"),
			"schema": json_schema,
			"strict": True,
			},
		}


class StructuredClient(BaseGroqClient, StructuredGenerationCapability):
	"""
	Stateless client for JSON-mode / schema-constrained generation.

	Usage
	-----
	Plain JSON mode (model writes valid JSON, no schema enforced):
		response = client.invoke_structured(
			..., json_schema=None
		)
		data = json.loads(response.text)

	Schema-constrained (model output must match the provided JSON Schema dict):
		schema = Person.model_json_schema()   # from a Pydantic model
		response = client.invoke_structured(
			..., json_schema=schema
		)
		person = Person.model_validate_json(response.text)
	"""

	# ------------------------------------------------------------------
	# Sync
	# ------------------------------------------------------------------

	def invoke_structured(
			self,
			api_key: str,
			model: str,
			messages: list[dict[str, str]],
			config: RequestConfig,
			session_id: str,
			api_key_id: str,
			json_schema: dict[str, Any] | None = None,
			request_id: str | None = None,
			) -> GroqResponse:
		"""
		Blocking structured completion.

		Returns a GroqResponse whose .text field contains valid JSON.
		The caller is responsible for parsing / validating it.
		"""
		rid = _new_rid(request_id)
		client = self._sync_sdk(api_key)
		response_format = _build_response_format(json_schema)
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
				response_format=response_format,  # type: ignore[arg-type]
				stream=False,
				)
		except Exception as exc:
			self._handle_sdk_error(exc, rid, api_key_id)

		choice = _first_choice(completion, rid)
		latency = time.perf_counter() - start
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

	async def async_invoke_structured(
			self,
			api_key: str,
			model: str,
			messages: list[dict[str, str]],
			config: RequestConfig,
			session_id: str,
			api_key_id: str,
			json_schema: dict[str, Any] | None = None,
			request_id: str | None = None,
			) -> GroqResponse:
		"""
		Async structured completion.

		Identical contract to invoke_structured() but uses the native async
		Groq SDK path.
		"""
		rid = _new_rid(request_id)
		client = self._async_sdk(api_key)
		response_format = _build_response_format(json_schema)
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
				response_format=response_format,  # type: ignore[arg-type]
				stream=False,
				)
		except Exception as exc:
			self._handle_sdk_error(exc, rid, api_key_id)

		choice = _first_choice(completion, rid)
		latency = time.perf_counter() - start
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
