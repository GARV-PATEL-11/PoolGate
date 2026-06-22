"""
chat_client.py — ChatClient

Implements TextGenerationCapability for all conversational / instruction-following
models in the PoolGate registry.

Supported models:
  allam-2-7b
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
  invoke()   — blocking single-turn completion
  async_invoke()  — async single-turn completion
  stream()   — blocking token-by-token generator
  async_stream()  — async token-by-token async generator
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Callable, Generator

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
from clients.capabilities import TextGenerationCapability
from schemas.runtime import GroqResponse, RequestConfig, TokenUsage


class ChatClient(BaseGroqClient, TextGenerationCapability):
	"""
	Stateless client for text generation (chat completion).

	Both the sync and async paths live here so callers never need to import
	two different classes.  Key rotation is handled upstream by the scheduler —
	this class receives the raw API key per call.
	"""

	# ------------------------------------------------------------------
	# Sync
	# ------------------------------------------------------------------

	def invoke(
			self,
			api_key: str,
			model: str,
			messages: list[dict[str, str]],
			config: RequestConfig,
			session_id: str,
			api_key_id: str,
			request_id: str | None = None,
			) -> GroqResponse:
		"""
		Blocking chat completion.

		Returns a GroqResponse with the full completion text, token usage,
		finish reason, and latency.
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
				stream=False,
				)
		except Exception as exc:
			self._handle_sdk_error(exc, rid, api_key_id)
			raise

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

	def stream(
			self,
			api_key: str,
			model: str,
			messages: list[dict[str, str]],
			config: RequestConfig,
			session_id: str,
			api_key_id: str,
			request_id: str | None = None,
			on_usage: Callable[[TokenUsage], None] | None = None,
			) -> Generator[str, None, None]:
		"""
		Blocking token-by-token generator.

		Yields each text delta as it arrives from the Groq streaming endpoint.

		Groq's SDK exposes streaming via chat.completions.create(stream=True),
		which returns an iterable Stream[ChatCompletionChunk] — there is no
		separate chat.completions.stream() context-manager method (verified
		against groq==1.4.0; that API exists in some other SDKs but not here).

		Token usage is not part of the OpenAI-style stream_options mechanism
		in this SDK version. Groq instead attaches usage to the final chunk
		as a vendor extension (chunk.x_groq.usage). If `on_usage` is given,
		it is invoked once with the parsed TokenUsage as soon as the final
		chunk carrying it is seen.
		"""
		rid = _new_rid(request_id)
		client = self._sync_sdk(api_key)

		try:
			stream = client.chat.completions.create(
				model=model,
				messages=messages,  # type: ignore[arg-type]
				temperature=config.temperature,
				top_p=config.top_p,
				max_tokens=config.max_tokens,
				seed=config.seed,
				stop=config.stop,
				timeout=config.timeout,
				stream=True,
				)
			with stream as s:
				for chunk in s:
					usage = _parse_chunk_usage(chunk)
					if usage is not None and on_usage is not None:
						on_usage(usage)
					delta = _chunk_delta_text(chunk, rid)
					if delta:
						yield delta
		except Exception as exc:
			self._handle_sdk_error(exc, rid, api_key_id)
			raise

	# ------------------------------------------------------------------
	# Async
	# ------------------------------------------------------------------

	async def async_invoke(
			self,
			api_key: str,
			model: str,
			messages: list[dict[str, str]],
			config: RequestConfig,
			session_id: str,
			api_key_id: str,
			request_id: str | None = None,
			) -> GroqResponse:
		"""
		Async chat completion.

		Identical contract to invoke() but uses the native async Groq SDK -
		never wraps a sync call in run_in_executor.
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
				stream=False,
				)
		except Exception as exc:
			self._handle_sdk_error(exc, rid, api_key_id)
			raise

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

	async def async_stream(
			self,
			api_key: str,
			model: str,
			messages: list[dict[str, str]],
			config: RequestConfig,
			session_id: str,
			api_key_id: str,
			request_id: str | None = None,
			on_usage: Callable[[TokenUsage], None] | None = None,
			) -> AsyncGenerator[str, None]:
		"""
		Async token-by-token generator.

		Yields each text delta as it arrives.  See stream()'s docstring for
		the same notes on the real SDK's streaming API shape and how usage
		is surfaced via the optional on_usage callback.
		"""
		rid = _new_rid(request_id)
		client = self._async_sdk(api_key)

		try:
			stream = await client.chat.completions.create(
				model=model,
				messages=messages,  # type: ignore[arg-type]
				temperature=config.temperature,
				top_p=config.top_p,
				max_tokens=config.max_tokens,
				seed=config.seed,
				stop=config.stop,
				timeout=config.timeout,
				stream=True,
				)
			async with stream as s:
				async for chunk in s:
					usage = _parse_chunk_usage(chunk)
					if usage is not None and on_usage is not None:
						on_usage(usage)
					delta = _chunk_delta_text(chunk, rid)
					if delta:
						yield delta
		except Exception as exc:
			self._handle_sdk_error(exc, rid, api_key_id)
			raise
