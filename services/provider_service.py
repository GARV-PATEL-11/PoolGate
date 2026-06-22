"""Primary PoolGate provider facade."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import (
	TYPE_CHECKING,
	Any,
	BinaryIO,
	TypeVar,
	)

from pydantic import BaseModel, ValidationError

from clients import (
	ChatClient,
	ModerationClient,
	ModerationResult,
	StructuredClient,
	SynthesisClient,
	SynthesisResult,
	ToolClient,
	TranscriptionClient,
	TranscriptionResult,
	assert_capability,
	)
from core.config import GroqConfig
from core.logger_manager import LoggerManager
from exceptions.configuration import EmptyKeyPoolError
from exceptions.keys import APIKeyDisabledError, NoAvailableAPIKeyError
from exceptions.output import SessionExpiredError, StructuredOutputError
from exceptions.request import MissingPromptError
from exceptions.response import RetryExhaustedError
from key_manager.key_pool import APIKeyState
from core.logger_manager import RequestContext
from retry import _is_auth_error, _is_rate_limit
from schedulers.request_scheduler import RequestScheduler
from schemas.runtime import (
	BatchResult,
	BatchSummary,
	GroqResponse,
	RequestConfig,
	RuntimeChatMessage,
	TokenUsage,
	)
from services.health_service import HealthService
from services.session_service import SessionManager
from tracking.manager import TrackingManager

from services.persistence_service import PersistenceService, RequestJournal


T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MODERATION_MODEL = "meta-llama/llama-prompt-guard-2-86m"
DEFAULT_TRANSCRIPTION_MODEL = "whisper-large-v3"
DEFAULT_SYNTHESIS_MODEL = "canopylabs/orpheus-v1-english"


def _prompt_to_messages(prompt: str, system: str | None = None) -> list[dict[str, str]]:
	messages: list[dict[str, str]] = []
	if system:
		messages.append({"role": "system", "content": system})
	messages.append({"role": "user", "content": prompt})
	return messages


def _validate_messages(messages: list[dict[str, str]]) -> None:
	"""
	Validate caller-supplied messages before they reach the Groq SDK.

	Raises InvalidMessageRoleError (via RuntimeChatMessage.validate_role)
	for any message with an unrecognised role. Messages built internally by
	_prompt_to_messages() are always well-formed and don't need to pass
	through here, but anything a caller hands to chat()/async_chat()/
	stream()/async_stream() directly does.
	"""
	for message in messages:
		RuntimeChatMessage(role=message.get("role", ""), content=message.get("content", ""))


def _extract_json(text: str) -> str:
	text = text.strip()
	if text.startswith("```"):
		lines = text.splitlines()
		text = "\n".join(line for line in lines[1:] if line.strip() != "```").strip()
	return text


def _sanitize_parse_error(exc: Exception, *, max_len: int = 200) -> str:
	"""
	Produce a short, safe-to-reprompt summary of a JSON parse failure.

	The raw exception string can echo back substrings of the model's own
	prior output (e.g. pydantic's ValidationError repeats the offending
	value). Feeding that unbounded text into a fresh user-role message as
	part of the JSON-repair retry loop is a narrow but real prompt-injection
	surface — truncate hard and strip newlines so it reads as a short error
	code, not as freeform text an adversarial completion could shape.
	"""
	message = str(exc).replace("\n", " ").replace("\r", " ").strip()
	if len(message) > max_len:
		message = message[:max_len].rstrip() + "…"
	return message


class GroqService:
	"""Production-facing Groq facade with key scheduling and session tracking."""

	def __init__(
			self,
			config: GroqConfig | None = None,
			session_id: str | None = None,
			debug_mode: bool = False,
			persistence: PersistenceService | None = None,
			) -> None:
		"""
		persistence: optional PersistenceService used for all three trackers.
			When None and POOLGATE_DATA_DIR is set, separate per-tracker JSON
			files are auto-created inside that directory so data survives restarts
			without any explicit configuration.  Pass an explicit PersistenceService
			to override the auto-detected location.
		"""
		self._config = config or GroqConfig.from_env()
		self._debug = debug_mode or self._config.debug_mode

		# All logging goes through LoggerManager; all paths come from PathConfig.
		self._log_manager = LoggerManager(
			self._config.paths,
			level=self._config.log_level,
			debug=self._debug,
			)
		self._logger = self._log_manager.get("service")

		self._session_manager = SessionManager(self._config.session_ttl_hours)
		self._tracking = TrackingManager()
		self._usage_tracker = self._tracking.usage_tracker
		self._health_service = HealthService()

		paths = self._config.paths

		# Per-tracker persistence — each tracker gets its own file so their
		# serialised formats never collide.
		if persistence is not None:
			# Explicit override: share one backend for all trackers (legacy path).
			self._usage_persistence: PersistenceService | None = persistence
			self._token_persistence: PersistenceService | None = persistence
			self._account_persistence: PersistenceService | None = persistence
		elif paths.tracking_dir and paths.usage_json and paths.tokens_json and paths.account_json:
			paths.ensure_dirs()
			self._usage_persistence = PersistenceService.json(paths.usage_json)
			self._token_persistence = PersistenceService.json(paths.tokens_json)
			self._account_persistence = PersistenceService.json(paths.account_json)
		else:
			self._usage_persistence = None
			self._token_persistence = None
			self._account_persistence = None

		if self._usage_persistence is not None:
			self._usage_persistence.load_tracker(self._tracking.usage_tracker)
			self._token_persistence.load_tracker(self._tracking.token_tracker)  # type: ignore[union-attr]
			self._account_persistence.load_tracker(self._tracking.account_tracker)  # type: ignore[union-attr]
			self._log_manager.log_storage(
				event="load", tracker="all",
				path=paths.tracking_dir or "",
				)
			self._logger.info("GroqService loaded tracker history from persistence backend.")

		# Request journal — one JSONL file per day with full execution details.
		self._journal: RequestJournal | None = (
			RequestJournal(paths.requests_dir)
			if paths.requests_dir
			else None
		)

		keys = [
			APIKeyState.from_key(
				key_id=f"key_{index}",
				raw_key=key,
				max_parallel=self._config.max_active_requests,
				)
			for index, key in enumerate(self._config.api_keys)
			]
		if not keys:
			raise EmptyKeyPoolError("GroqService requires at least one API key.")
		self._scheduler = RequestScheduler(keys, self._config, self._logger)
		self._chat_client = ChatClient()
		self._structured_client = StructuredClient()
		self._tool_client = ToolClient()
		self._moderation_client = ModerationClient()
		self._transcription_client = TranscriptionClient()
		self._synthesis_client = SynthesisClient()
		self._default_session_id = session_id

		self._logger.info(
			f"GroqService initialized with {len(keys)} API key(s). debug={self._debug}",
			)
		self._log_manager.log_trace(
			rid="init", stage="initialized",
			keys=len(keys), debug=self._debug,
			persistence=paths.persistence_enabled,
			)

	def flush_tracking(self) -> None:
		"""
		Persist current tracker state to the configured JSON files.
		No-op if no persistence backend was configured.  Call this
		periodically or before process shutdown to avoid losing in-memory data.
		"""
		paths = self._config.paths
		if self._usage_persistence is not None:
			self._usage_persistence.flush_tracker(self._tracking.usage_tracker)
			self._log_manager.log_storage(
				event="flush", tracker="usage", path=paths.usage_json or "",
				)
		if self._token_persistence is not None:
			self._token_persistence.flush_tracker(self._tracking.token_tracker)
			self._log_manager.log_storage(
				event="flush", tracker="tokens", path=paths.tokens_json or "",
				)
		if self._account_persistence is not None:
			self._account_persistence.flush_tracker(self._tracking.account_tracker)
			self._log_manager.log_storage(
				event="flush", tracker="account", path=paths.account_json or "",
				)

	def _resolve_session(self, session_id: str | None, request_id: str) -> str:
		try:
			session = self._session_manager.get_or_create(session_id or self._default_session_id)
		except SessionExpiredError as exc:
			raise SessionExpiredError(exc.session_id, request_id=request_id) from exc
		return session.session_id

	def _journal_entry(
			self,
			*,
			request_id: str,
			session_id: str,
			model: str,
			api_key_id: str,
			latency: float,
			tokens_in: int,
			tokens_out: int,
			success: bool,
			retried: bool,
			error: str | None = None,
			) -> None:
		if self._journal is None:
			return
		self._journal.append({
			"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
			"request_id": request_id,
			"session_id": session_id,
			"model": model,
			"api_key_id": api_key_id,
			"prompt_tokens": tokens_in,
			"completion_tokens": tokens_out,
			"total_tokens": tokens_in + tokens_out,
			"latency_seconds": round(latency, 4),
			"success": success,
			"retried": retried,
			"error": error,
			},
			)

	def _record_success(
			self,
			session_id: str,
			model: str,
			response: GroqResponse,
			latency: float,
			retried: bool,
			) -> None:
		self._tracking.record_success(
			model,
			tokens_in=response.usage.prompt_tokens,
			tokens_out=response.usage.completion_tokens,
			api_key_id=response.api_key_id,
			retried=retried,
			)
		session = self._session_manager.get(session_id)
		if session:
			session.record_success(
				model=model,
				tokens_in=response.usage.prompt_tokens,
				tokens_out=response.usage.completion_tokens,
				latency=latency,
				retried=retried,
				)
		self._journal_entry(
			request_id=response.request_id,
			session_id=session_id,
			model=model,
			api_key_id=response.api_key_id,
			latency=latency,
			tokens_in=response.usage.prompt_tokens,
			tokens_out=response.usage.completion_tokens,
			success=True,
			retried=retried,
			)

	def _record_failure(self, session_id: str, model: str, retried: bool) -> None:
		self._tracking.record_failure(model, retried=retried)
		session = self._session_manager.get(session_id)
		if session:
			session.record_failure(retried=retried)

	def _record_success_raw(
			self,
			session_id: str,
			model: str,
			latency: float,
			retried: bool,
			tokens_in: int = 0,
			tokens_out: int = 0,
			api_key_id: str = "",
			request_id: str = "",
			) -> None:
		"""Record success for non-GroqResponse results (moderation, transcription, synthesis)."""
		self._tracking.record_success(
			model,
			tokens_in=tokens_in,
			tokens_out=tokens_out,
			api_key_id=api_key_id,
			retried=retried,
			)
		session = self._session_manager.get(session_id)
		if session:
			session.record_success(
				model=model,
				tokens_in=tokens_in,
				tokens_out=tokens_out,
				latency=latency,
				retried=retried,
				)
		self._journal_entry(
			request_id=request_id,
			session_id=session_id,
			model=model,
			api_key_id=api_key_id,
			latency=latency,
			tokens_in=tokens_in,
			tokens_out=tokens_out,
			success=True,
			retried=retried,
			)

	def _run_rotation_generic(
			self,
			model: str,
			request_id: str,
			session_id: str,
			config: RequestConfig,
			call: Callable[[APIKeyState], Any],
			tokens_extractor: Callable[[Any], tuple[int, int]] | None = None,
			capability: str = "api",
			) -> Any:
		"""Key-rotating dispatch for non-GroqResponse results (moderation, transcription, synthesis)."""
		ctx = RequestContext(request_id=request_id, session_id=session_id, model=model)
		attempts = config.retries + 1
		last_exc: Exception | None = None

		self._log_manager.log_request(
			rid=request_id, sid=session_id, model=model,
			capability=capability, attempt=1,
			)

		for attempt in range(attempts):
			key: APIKeyState | None = None
			start = time.perf_counter()
			try:
				key = self._scheduler.acquire_key(request_id, model=model)
				ctx.api_key_id = key.key_id
				ctx.retry_count = attempt
				self._log_manager.log_trace(
					rid=request_id, stage="key_acquired",
					key_id=key.key_id, attempt=attempt + 1,
					)
				result = call(key)
				latency = time.perf_counter() - start
				tokens_in, tokens_out = tokens_extractor(result) if tokens_extractor else (0, 0)
				self._scheduler.release_key(
					key,
					latency=latency,
					tokens_in=tokens_in,
					tokens_out=tokens_out,
					)
				self._log_manager.log_response(
					rid=request_id, model=model,
					tokens_in=tokens_in, tokens_out=tokens_out,
					latency=latency, success=True,
					)
				self._log_manager.log_performance(
					rid=request_id, model=model, latency=latency,
					tokens_in=tokens_in, tokens_out=tokens_out,
					)
				self._record_success_raw(
					session_id,
					model,
					latency,
					retried=attempt > 0,
					tokens_in=tokens_in,
					tokens_out=tokens_out,
					api_key_id=key.key_id,
					request_id=request_id,
					)
				return result
			except (NoAvailableAPIKeyError, APIKeyDisabledError):
				if key:
					self._scheduler.mark_key_disabled(key)
				raise
			except Exception as exc:
				last_exc = exc
				is_auth = _is_auth_error(exc)
				if key:
					if is_auth:
						self._scheduler.mark_key_disabled(key)
					else:
						self._scheduler.mark_key_failure(key, is_rate_limit=_is_rate_limit(exc))
				self._logger.warning(
					f"Attempt {attempt + 1}/{attempts} failed: {type(exc).__name__}: {exc}",
					ctx,
					)
				self._log_manager.log_trace(
					rid=request_id, stage="retry",
					attempt=attempt + 1, error=type(exc).__name__,
					)
				if is_auth:
					break

		self._log_manager.log_response(
			rid=request_id, model=model,
			tokens_in=0, tokens_out=0, latency=0.0,
			success=False, error=type(last_exc).__name__ if last_exc else "unknown",
			)
		self._log_manager.log_trace(
			rid=request_id, stage="exhausted", attempts=attempts,
			)
		self._record_failure(session_id, model, retried=attempts > 1)
		raise RetryExhaustedError(
			f"All {attempts} attempt(s) failed. Last error: {last_exc}",
			attempts=attempts,
			last_exc=last_exc,
			request_id=request_id,
			) from last_exc

	async def _async_run_rotation_generic(
			self,
			model: str,
			request_id: str,
			session_id: str,
			config: RequestConfig,
			call: Callable[[APIKeyState], Awaitable[Any]],
			tokens_extractor: Callable[[Any], tuple[int, int]] | None = None,
			capability: str = "api",
			) -> Any:
		"""Async key-rotating dispatch for non-GroqResponse results."""
		ctx = RequestContext(request_id=request_id, session_id=session_id, model=model)
		attempts = config.retries + 1
		last_exc: Exception | None = None

		self._log_manager.log_request(
			rid=request_id, sid=session_id, model=model,
			capability=capability, attempt=1,
			)

		for attempt in range(attempts):
			key: APIKeyState | None = None
			start = time.perf_counter()
			try:
				key = await self._scheduler.async_acquire_key(request_id, model=model)
				ctx.api_key_id = key.key_id
				ctx.retry_count = attempt
				self._log_manager.log_trace(
					rid=request_id, stage="key_acquired",
					key_id=key.key_id, attempt=attempt + 1,
					)
				result = await call(key)
				latency = time.perf_counter() - start
				tokens_in, tokens_out = tokens_extractor(result) if tokens_extractor else (0, 0)
				self._scheduler.release_key(
					key,
					latency=latency,
					tokens_in=tokens_in,
					tokens_out=tokens_out,
					)
				self._log_manager.log_response(
					rid=request_id, model=model,
					tokens_in=tokens_in, tokens_out=tokens_out,
					latency=latency, success=True,
					)
				self._log_manager.log_performance(
					rid=request_id, model=model, latency=latency,
					tokens_in=tokens_in, tokens_out=tokens_out,
					)
				self._record_success_raw(
					session_id,
					model,
					latency,
					retried=attempt > 0,
					tokens_in=tokens_in,
					tokens_out=tokens_out,
					api_key_id=key.key_id,
					request_id=request_id,
					)
				return result
			except (NoAvailableAPIKeyError, APIKeyDisabledError):
				if key:
					self._scheduler.mark_key_disabled(key)
				raise
			except Exception as exc:
				last_exc = exc
				is_auth = _is_auth_error(exc)
				if key:
					if is_auth:
						self._scheduler.mark_key_disabled(key)
					else:
						self._scheduler.mark_key_failure(key, is_rate_limit=_is_rate_limit(exc))
				self._logger.warning(
					f"Async attempt {attempt + 1}/{attempts} failed: {type(exc).__name__}: {exc}",
					ctx,
					)
				self._log_manager.log_trace(
					rid=request_id, stage="retry",
					attempt=attempt + 1, error=type(exc).__name__,
					)
				if is_auth:
					break

		self._log_manager.log_response(
			rid=request_id, model=model,
			tokens_in=0, tokens_out=0, latency=0.0,
			success=False, error=type(last_exc).__name__ if last_exc else "unknown",
			)
		self._log_manager.log_trace(
			rid=request_id, stage="exhausted", attempts=attempts,
			)
		self._record_failure(session_id, model, retried=attempts > 1)
		raise RetryExhaustedError(
			f"All {attempts} async attempt(s) failed. Last error: {last_exc}",
			attempts=attempts,
			last_exc=last_exc,
			request_id=request_id,
			) from last_exc

	def _run_with_rotation(
			self,
			model: str,
			request_id: str,
			session_id: str,
			config: RequestConfig,
			call: Callable[[APIKeyState], GroqResponse],
			capability: str = "chat",
			) -> GroqResponse:
		"""
		Retry loop with per-attempt key rotation.

		Deliberately NOT built on retry.py's RetryPolicy/RetryService: those
		retry the *same* call with a fixed key and exponential backoff, which
		is the right model when there's only one upstream identity to retry
		against. Here, each retry should acquire a *different* key from the
		pool rather than backing off and re-hitting the same one — if key A
		just got rate-limited, immediately trying key B is strictly better
		than waiting and retrying key A. RetryService remains available for
		call sites that don't rotate keys (e.g. a single fixed-key retry).
		"""
		ctx = RequestContext(request_id=request_id, session_id=session_id, model=model)
		attempts = config.retries + 1
		last_exc: Exception | None = None

		self._log_manager.log_request(
			rid=request_id, sid=session_id, model=model,
			capability=capability, attempt=1,
			)

		for attempt in range(attempts):
			key: APIKeyState | None = None
			start = time.perf_counter()
			try:
				key = self._scheduler.acquire_key(request_id, model=model)
				ctx.api_key_id = key.key_id
				ctx.retry_count = attempt
				self._log_manager.log_trace(
					rid=request_id, stage="key_acquired",
					key_id=key.key_id, attempt=attempt + 1,
					)
				response = call(key)
				latency = time.perf_counter() - start
				self._scheduler.release_key(
					key,
					latency=latency,
					tokens_in=response.usage.prompt_tokens,
					tokens_out=response.usage.completion_tokens,
					)
				self._log_manager.log_response(
					rid=request_id, model=model,
					tokens_in=response.usage.prompt_tokens,
					tokens_out=response.usage.completion_tokens,
					latency=latency, success=True,
					finish_reason=response.finish_reason.value if response.finish_reason else "",
					)
				self._log_manager.log_performance(
					rid=request_id, model=model, latency=latency,
					tokens_in=response.usage.prompt_tokens,
					tokens_out=response.usage.completion_tokens,
					)
				self._record_success(session_id, model, response, latency, retried=attempt > 0)
				return response
			except (NoAvailableAPIKeyError, APIKeyDisabledError):
				if key:
					self._scheduler.mark_key_disabled(key)
				raise
			except Exception as exc:
				last_exc = exc
				latency = time.perf_counter() - start
				is_auth = _is_auth_error(exc)
				if key:
					if is_auth:
						self._scheduler.mark_key_disabled(key)
					else:
						self._scheduler.mark_key_failure(key, is_rate_limit=_is_rate_limit(exc))
				self._logger.warning(
					f"Attempt {attempt + 1}/{attempts} failed: {type(exc).__name__}: {exc}",
					ctx,
					)
				self._log_manager.log_trace(
					rid=request_id, stage="retry",
					attempt=attempt + 1, error=type(exc).__name__,
					)
				if is_auth:
					break

		self._log_manager.log_response(
			rid=request_id, model=model,
			tokens_in=0, tokens_out=0, latency=0.0,
			success=False, error=type(last_exc).__name__ if last_exc else "unknown",
			)
		self._log_manager.log_trace(
			rid=request_id, stage="exhausted", attempts=attempts,
			)
		self._record_failure(session_id, model, retried=attempts > 1)
		raise RetryExhaustedError(
			f"All {attempts} attempt(s) failed. Last error: {last_exc}",
			attempts=attempts,
			last_exc=last_exc,
			request_id=request_id,
			) from last_exc

	async def _async_run_with_rotation(
			self,
			model: str,
			request_id: str,
			session_id: str,
			config: RequestConfig,
			call: Callable[[APIKeyState], object],
			capability: str = "chat",
			) -> GroqResponse:
		ctx = RequestContext(request_id=request_id, session_id=session_id, model=model)
		attempts = config.retries + 1
		last_exc: Exception | None = None

		self._log_manager.log_request(
			rid=request_id, sid=session_id, model=model,
			capability=capability, attempt=1,
			)

		for attempt in range(attempts):
			key: APIKeyState | None = None
			start = time.perf_counter()
			try:
				key = await self._scheduler.async_acquire_key(request_id, model=model)
				ctx.api_key_id = key.key_id
				ctx.retry_count = attempt
				self._log_manager.log_trace(
					rid=request_id, stage="key_acquired",
					key_id=key.key_id, attempt=attempt + 1,
					)
				response = await call(key)  # type: ignore[misc]
				latency = time.perf_counter() - start
				self._scheduler.release_key(
					key,
					latency=latency,
					tokens_in=response.usage.prompt_tokens,
					tokens_out=response.usage.completion_tokens,
					)
				self._log_manager.log_response(
					rid=request_id, model=model,
					tokens_in=response.usage.prompt_tokens,
					tokens_out=response.usage.completion_tokens,
					latency=latency, success=True,
					finish_reason=response.finish_reason.value if response.finish_reason else "",
					)
				self._log_manager.log_performance(
					rid=request_id, model=model, latency=latency,
					tokens_in=response.usage.prompt_tokens,
					tokens_out=response.usage.completion_tokens,
					)
				self._record_success(session_id, model, response, latency, retried=attempt > 0)
				return response
			except (NoAvailableAPIKeyError, APIKeyDisabledError):
				if key:
					self._scheduler.mark_key_disabled(key)
				raise
			except Exception as exc:
				last_exc = exc
				is_auth = _is_auth_error(exc)
				if key:
					if is_auth:
						self._scheduler.mark_key_disabled(key)
					else:
						self._scheduler.mark_key_failure(key, is_rate_limit=_is_rate_limit(exc))
				self._logger.warning(
					f"Async attempt {attempt + 1}/{attempts} failed: {type(exc).__name__}: {exc}",
					ctx,
					)
				self._log_manager.log_trace(
					rid=request_id, stage="retry",
					attempt=attempt + 1, error=type(exc).__name__,
					)
				if is_auth:
					break

		self._log_manager.log_response(
			rid=request_id, model=model,
			tokens_in=0, tokens_out=0, latency=0.0,
			success=False, error=type(last_exc).__name__ if last_exc else "unknown",
			)
		self._log_manager.log_trace(
			rid=request_id, stage="exhausted", attempts=attempts,
			)
		self._record_failure(session_id, model, retried=attempts > 1)
		raise RetryExhaustedError(
			f"All {attempts} async attempt(s) failed. Last error: {last_exc}",
			attempts=attempts,
			last_exc=last_exc,
			request_id=request_id,
			) from last_exc

	def invoke(
			self,
			prompt: str,
			model: str = DEFAULT_MODEL,
			system: str | None = None,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> GroqResponse:
		assert_capability(model, "chat")
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		messages = _prompt_to_messages(prompt, system)
		return self._run_with_rotation(
			model,
			rid,
			sid,
			cfg,
			lambda key: self._chat_client.invoke(
				api_key=key.raw_key,
				model=model,
				messages=messages,
				config=cfg,
				session_id=sid,
				api_key_id=key.key_id,
				request_id=rid,
				),
			)

	async def async_invoke(
			self,
			prompt: str,
			model: str = DEFAULT_MODEL,
			system: str | None = None,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> GroqResponse:
		assert_capability(model, "chat")
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		messages = _prompt_to_messages(prompt, system)
		return await self._async_run_with_rotation(
			model,
			rid,
			sid,
			cfg,
			lambda key: self._chat_client.async_invoke(
				api_key=key.raw_key,
				model=model,
				messages=messages,
				config=cfg,
				session_id=sid,
				api_key_id=key.key_id,
				request_id=rid,
				),
			)

	def chat(
			self,
			messages: list[dict[str, str]],
			model: str = DEFAULT_MODEL,
			temperature: float = 1.0,
			top_p: float = 1.0,
			max_tokens: int | None = None,
			seed: int | None = None,
			stop: list[str] | str | None = None,
			timeout: float = 30.0,
			retries: int | None = None,
			session_id: str | None = None,
			) -> GroqResponse:
		cfg = RequestConfig(
			temperature=temperature,
			top_p=top_p,
			max_tokens=max_tokens,
			seed=seed,
			stop=stop,
			timeout=timeout,
			retries=retries if retries is not None else self._config.max_retries,
			)
		return self._chat(messages, model, cfg, session_id)

	def _chat(
			self,
			messages: list[dict[str, str]],
			model: str,
			config: RequestConfig,
			session_id: str | None,
			) -> GroqResponse:
		assert_capability(model, "chat")
		_validate_messages(messages)
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		return self._run_with_rotation(
			model,
			rid,
			sid,
			config,
			lambda key: self._chat_client.invoke(
				api_key=key.raw_key,
				model=model,
				messages=messages,
				config=config,
				session_id=sid,
				api_key_id=key.key_id,
				request_id=rid,
				),
			)

	async def async_chat(
			self,
			messages: list[dict[str, str]],
			model: str = DEFAULT_MODEL,
			temperature: float = 1.0,
			top_p: float = 1.0,
			max_tokens: int | None = None,
			seed: int | None = None,
			stop: list[str] | str | None = None,
			timeout: float = 30.0,
			retries: int | None = None,
			session_id: str | None = None,
			) -> GroqResponse:
		assert_capability(model, "chat")
		_validate_messages(messages)
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = RequestConfig(
			temperature=temperature,
			top_p=top_p,
			max_tokens=max_tokens,
			seed=seed,
			stop=stop,
			timeout=timeout,
			retries=retries if retries is not None else self._config.max_retries,
			)
		return await self._async_run_with_rotation(
			model,
			rid,
			sid,
			cfg,
			lambda key: self._chat_client.async_invoke(
				api_key=key.raw_key,
				model=model,
				messages=messages,
				config=cfg,
				session_id=sid,
				api_key_id=key.key_id,
				request_id=rid,
				),
			)

	def structured(
			self,
			prompt: str,
			schema: type[T],
			model: str = DEFAULT_MODEL,
			system: str | None = None,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			json_retries: int = 2,
			) -> T:
		assert_capability(model, "structured")
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		messages = _prompt_to_messages(prompt, system)
		json_schema = schema.model_json_schema()
		last_exc: Exception | None = None
		response: GroqResponse | None = None

		for _ in range(json_retries + 1):
			response = self._run_with_rotation(
				model,
				rid,
				sid,
				cfg,
				lambda key, _msgs=messages: self._structured_client.invoke_structured(  # noqa: B023
					api_key=key.raw_key,
					model=model,
					messages=_msgs,
					config=cfg,
					session_id=sid,
					api_key_id=key.key_id,
					json_schema=json_schema,
					request_id=rid,
					),
				capability="structured",
				)
			try:
				return schema.model_validate_json(_extract_json(response.text))
			except (ValidationError, json.JSONDecodeError, ValueError) as exc:
				last_exc = exc
				messages = messages + [
					{"role": "assistant", "content": response.text},
					{
						"role": "user",
						"content": f"Fix the JSON only. Parse error: {_sanitize_parse_error(exc)}",
						},
					]

		raise StructuredOutputError(
			f"Failed to parse structured output after {json_retries + 1} attempts: {last_exc}",
			raw_response=response.text if response else None,
			request_id=rid,
			)

	async def async_structured(
			self,
			prompt: str,
			schema: type[T],
			model: str = DEFAULT_MODEL,
			system: str | None = None,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			json_retries: int = 2,
			) -> T:
		assert_capability(model, "structured")
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		messages = _prompt_to_messages(prompt, system)
		json_schema = schema.model_json_schema()
		last_exc: Exception | None = None
		response: GroqResponse | None = None

		for _ in range(json_retries + 1):
			response = await self._async_run_with_rotation(
				model,
				rid,
				sid,
				cfg,
				lambda key, _msgs=messages: self._structured_client.async_invoke_structured(  # noqa: B023
					api_key=key.raw_key,
					model=model,
					messages=_msgs,
					config=cfg,
					session_id=sid,
					api_key_id=key.key_id,
					json_schema=json_schema,
					request_id=rid,
					),
				capability="structured",
				)
			try:
				return schema.model_validate_json(_extract_json(response.text))
			except (ValidationError, json.JSONDecodeError, ValueError) as exc:
				last_exc = exc
				messages += [
					{"role": "assistant", "content": response.text},
					{
						"role": "user",
						"content": f"Fix the JSON only. Parse error: {_sanitize_parse_error(exc)}",
						},
					]

		raise StructuredOutputError(
			f"Failed to parse async structured output after {json_retries + 1} attempts: {last_exc}",
			raw_response=response.text if response else None,
			request_id=rid,
			)

	def stream(
			self,
			prompt: str | None = None,
			messages: list[dict[str, str]] | None = None,
			model: str = DEFAULT_MODEL,
			system: str | None = None,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> Generator[str, None, None]:
		"""
		Blocking token-by-token generator.

		Retries are honored only before the first chunk has been yielded to
		the caller — once any text has been delivered, re-running the whole
		request would duplicate output the caller has already received, so
		a mid-stream failure after the first chunk is raised immediately
		rather than retried. config.retries (or self._config.max_retries by
		default) bounds the number of pre-first-chunk attempts.

		Token usage IS recorded for streaming calls (Groq attaches it to the
		final chunk as chunk.x_groq.usage — see clients/chat_client.py).
		"""
		assert_capability(model, "chat")
		rid = str(uuid.uuid4())
		if messages is None and prompt is None:
			raise MissingPromptError("Provide either 'prompt' or 'messages'.", request_id=rid)
		if messages is not None:
			_validate_messages(messages)
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(stream=True, retries=self._config.max_retries)
		payload = messages or _prompt_to_messages(prompt or "", system)

		attempts = cfg.retries + 1
		last_exc: Exception | None = None

		for attempt in range(attempts):
			key = self._scheduler.acquire_key(rid, model=model)
			start = time.perf_counter()
			usage_holder: dict[str, TokenUsage] = {}
			first_chunk_yielded = False
			try:
				for delta in self._chat_client.stream(
						api_key=key.raw_key,
						model=model,
						messages=payload,
						config=cfg,
						session_id=sid,
						api_key_id=key.key_id,
						request_id=rid,
						on_usage=lambda u, _h=usage_holder: _h.__setitem__("usage", u),  # noqa: B023
						):
					first_chunk_yielded = True
					yield delta
				usage = usage_holder.get("usage", TokenUsage())
				self._scheduler.release_key(
					key,
					latency=time.perf_counter() - start,
					tokens_in=usage.prompt_tokens,
					tokens_out=usage.completion_tokens,
					)
				self._record_success(
					sid,
					model,
					GroqResponse(
						text="",
						model=model,
						usage=usage,
						latency=time.perf_counter() - start,
						session_id=sid,
						request_id=rid,
						api_key_id=key.key_id,
						),
					time.perf_counter() - start,
					retried=attempt > 0,
					)
				return
			except Exception as exc:
				last_exc = exc
				self._scheduler.mark_key_failure(key, is_rate_limit=_is_rate_limit(exc))
				if first_chunk_yielded:
					# Output already delivered to the caller — retrying now
					# would duplicate it. Surface the failure immediately.
					raise
				self._logger.warning(
					f"Stream attempt {attempt + 1}/{attempts} failed before "
					f"first chunk: {type(exc).__name__}: {exc}",
					)

		self._record_failure(sid, model, retried=attempts > 1)
		raise RetryExhaustedError(
			f"All {attempts} stream attempt(s) failed before producing output. Last error: {last_exc}",
			attempts=attempts,
			last_exc=last_exc,
			request_id=rid,
			) from last_exc

	async def async_stream(
			self,
			prompt: str | None = None,
			messages: list[dict[str, str]] | None = None,
			model: str = DEFAULT_MODEL,
			system: str | None = None,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> AsyncGenerator[str, None]:
		"""
		Async token-by-token generator. See stream()'s docstring for the
		retry-before-first-chunk and token-accounting behavior — identical
		semantics here, just async.
		"""
		assert_capability(model, "chat")
		rid = str(uuid.uuid4())
		if messages is None and prompt is None:
			raise MissingPromptError("Provide either 'prompt' or 'messages'.", request_id=rid)
		if messages is not None:
			_validate_messages(messages)
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(stream=True, retries=self._config.max_retries)
		payload = messages or _prompt_to_messages(prompt or "", system)

		attempts = cfg.retries + 1
		last_exc: Exception | None = None

		for attempt in range(attempts):
			key = await self._scheduler.async_acquire_key(rid, model=model)
			start = time.perf_counter()
			usage_holder: dict[str, TokenUsage] = {}
			first_chunk_yielded = False
			try:
				async for delta in self._chat_client.async_stream(
						api_key=key.raw_key,
						model=model,
						messages=payload,
						config=cfg,
						session_id=sid,
						api_key_id=key.key_id,
						request_id=rid,
						on_usage=lambda u, _h=usage_holder: _h.__setitem__("usage", u),  # noqa: B023
						):
					first_chunk_yielded = True
					yield delta
				usage = usage_holder.get("usage", TokenUsage())
				self._scheduler.release_key(
					key,
					latency=time.perf_counter() - start,
					tokens_in=usage.prompt_tokens,
					tokens_out=usage.completion_tokens,
					)
				self._record_success(
					sid,
					model,
					GroqResponse(
						text="",
						model=model,
						usage=usage,
						latency=time.perf_counter() - start,
						session_id=sid,
						request_id=rid,
						api_key_id=key.key_id,
						),
					time.perf_counter() - start,
					retried=attempt > 0,
					)
				return
			except Exception as exc:
				last_exc = exc
				self._scheduler.mark_key_failure(key, is_rate_limit=_is_rate_limit(exc))
				if first_chunk_yielded:
					raise
				self._logger.warning(
					f"Async stream attempt {attempt + 1}/{attempts} failed before "
					f"first chunk: {type(exc).__name__}: {exc}",
					)

		self._record_failure(sid, model, retried=attempts > 1)
		raise RetryExhaustedError(
			f"All {attempts} async stream attempt(s) failed before producing output. Last error: {last_exc}",
			attempts=attempts,
			last_exc=last_exc,
			request_id=rid,
			) from last_exc

	async def batch(
			self,
			prompts: list[str],
			model: str = DEFAULT_MODEL,
			system: str | None = None,
			config: RequestConfig | None = None,
			concurrency: int | None = None,
			session_id: str | None = None,
			) -> BatchSummary:
		cfg = config or RequestConfig(retries=self._config.max_retries)
		sem = asyncio.Semaphore(concurrency or self._config.default_concurrency)

		async def _single(index: int, prompt: str) -> BatchResult:
			async with sem:
				try:
					response = await self.async_invoke(prompt, model, system, cfg, session_id)
					return BatchResult(index=index, response=response, success=True)
				except Exception as exc:
					return BatchResult(index=index, error=str(exc), success=False)

		results = await asyncio.gather(*[_single(i, prompt) for i, prompt in enumerate(prompts)])
		results.sort(key=lambda result: result.index)
		total_usage = TokenUsage()
		total_latency = 0.0
		for result in results:
			if result.response:
				total_usage += result.response.usage
				total_latency += result.response.latency
		succeeded = sum(1 for result in results if result.success)
		return BatchSummary(
			total=len(results),
			succeeded=succeeded,
			failed=len(results) - succeeded,
			results=results,
			total_tokens=total_usage,
			total_latency=total_latency,
			)

	# ------------------------------------------------------------------
	# Tool calling
	# ------------------------------------------------------------------

	def invoke_tools(
			self,
			messages: list[dict[str, Any]],
			tools: list[dict[str, Any]],
			model: str = DEFAULT_MODEL,
			tool_choice: str | dict[str, Any] = "auto",
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> GroqResponse:
		assert_capability(model, "tools")
		_validate_messages(messages)
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		response = self._run_with_rotation(
			model,
			rid,
			sid,
			cfg,
			lambda key: self._tool_client.invoke_tools(
				api_key=key.raw_key,
				model=model,
				messages=messages,
				config=cfg,
				session_id=sid,
				api_key_id=key.key_id,
				tools=tools,
				tool_choice=tool_choice,
				request_id=rid,
				),
			capability="tools",
			)
		tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
		self._log_manager.log_tool_call(
			rid=rid, model=model, tool_names=tool_names,
			latency=response.latency,
			finish_reason=response.finish_reason.value if response.finish_reason else "",
			)
		return response

	async def async_invoke_tools(
			self,
			messages: list[dict[str, Any]],
			tools: list[dict[str, Any]],
			model: str = DEFAULT_MODEL,
			tool_choice: str | dict[str, Any] = "auto",
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> GroqResponse:
		assert_capability(model, "tools")
		_validate_messages(messages)
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		response = await self._async_run_with_rotation(
			model,
			rid,
			sid,
			cfg,
			lambda key: self._tool_client.async_invoke_tools(
				api_key=key.raw_key,
				model=model,
				messages=messages,
				config=cfg,
				session_id=sid,
				api_key_id=key.key_id,
				tools=tools,
				tool_choice=tool_choice,
				request_id=rid,
				),
			capability="tools",
			)
		tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
		self._log_manager.log_tool_call(
			rid=rid, model=model, tool_names=tool_names,
			latency=response.latency,
			finish_reason=response.finish_reason.value if response.finish_reason else "",
			)
		return response

	# ------------------------------------------------------------------
	# Moderation
	# ------------------------------------------------------------------

	def moderate(
			self,
			text: str,
			model: str = DEFAULT_MODERATION_MODEL,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> ModerationResult:
		assert_capability(model, "moderation")
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		return self._run_rotation_generic(
			model,
			rid,
			sid,
			cfg,
			lambda key: self._moderation_client.moderate(
				api_key=key.raw_key,
				model=model,
				text=text,
				config=cfg,
				session_id=sid,
				api_key_id=key.key_id,
				request_id=rid,
				),
			tokens_extractor=lambda r: (r.usage.prompt_tokens, r.usage.completion_tokens),
			capability="moderation",
			)

	async def async_moderate(
			self,
			text: str,
			model: str = DEFAULT_MODERATION_MODEL,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> ModerationResult:
		assert_capability(model, "moderation")
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		return await self._async_run_rotation_generic(
			model,
			rid,
			sid,
			cfg,
			lambda key: self._moderation_client.async_moderate(
				api_key=key.raw_key,
				model=model,
				text=text,
				config=cfg,
				session_id=sid,
				api_key_id=key.key_id,
				request_id=rid,
				),
			tokens_extractor=lambda r: (r.usage.prompt_tokens, r.usage.completion_tokens),
			capability="moderation",
			)

	# ------------------------------------------------------------------
	# Transcription
	# ------------------------------------------------------------------

	def transcribe(
			self,
			audio_file: BinaryIO | bytes | tuple[str, bytes],
			model: str = DEFAULT_TRANSCRIPTION_MODEL,
			language: str | None = None,
			prompt: str | None = None,
			response_format: str = "text",
			temperature: float = 0.0,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> TranscriptionResult:
		assert_capability(model, "transcription")
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		return self._run_rotation_generic(
			model,
			rid,
			sid,
			cfg,
			lambda key: self._transcription_client.transcribe(
				api_key=key.raw_key,
				model=model,
				audio_file=audio_file,
				session_id=sid,
				api_key_id=key.key_id,
				language=language,
				prompt=prompt,
				response_format=response_format,
				temperature=temperature,
				request_id=rid,
				),
			capability="transcription",
			)

	async def async_transcribe(
			self,
			audio_file: BinaryIO | bytes | tuple[str, bytes],
			model: str = DEFAULT_TRANSCRIPTION_MODEL,
			language: str | None = None,
			prompt: str | None = None,
			response_format: str = "text",
			temperature: float = 0.0,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> TranscriptionResult:
		assert_capability(model, "transcription")
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		return await self._async_run_rotation_generic(
			model,
			rid,
			sid,
			cfg,
			lambda key: self._transcription_client.async_transcribe(
				api_key=key.raw_key,
				model=model,
				audio_file=audio_file,
				session_id=sid,
				api_key_id=key.key_id,
				language=language,
				prompt=prompt,
				response_format=response_format,
				temperature=temperature,
				request_id=rid,
				),
			capability="transcription",
			)

	def translate(
			self,
			audio_file: BinaryIO | bytes | tuple[str, bytes],
			model: str = DEFAULT_TRANSCRIPTION_MODEL,
			prompt: str | None = None,
			response_format: str = "text",
			temperature: float = 0.0,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> TranscriptionResult:
		assert_capability(model, "transcription")
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		return self._run_rotation_generic(
			model,
			rid,
			sid,
			cfg,
			lambda key: self._transcription_client.translate(
				api_key=key.raw_key,
				model=model,
				audio_file=audio_file,
				session_id=sid,
				api_key_id=key.key_id,
				prompt=prompt,
				response_format=response_format,
				temperature=temperature,
				request_id=rid,
				),
			capability="translation",
			)

	async def async_translate(
			self,
			audio_file: BinaryIO | bytes | tuple[str, bytes],
			model: str = DEFAULT_TRANSCRIPTION_MODEL,
			prompt: str | None = None,
			response_format: str = "text",
			temperature: float = 0.0,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> TranscriptionResult:
		assert_capability(model, "transcription")
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		return await self._async_run_rotation_generic(
			model,
			rid,
			sid,
			cfg,
			lambda key: self._transcription_client.async_translate(
				api_key=key.raw_key,
				model=model,
				audio_file=audio_file,
				session_id=sid,
				api_key_id=key.key_id,
				prompt=prompt,
				response_format=response_format,
				temperature=temperature,
				request_id=rid,
				),
			capability="translation",
			)

	# ------------------------------------------------------------------
	# Synthesis
	# ------------------------------------------------------------------

	def synthesize(
			self,
			text: str,
			voice: str,
			model: str = DEFAULT_SYNTHESIS_MODEL,
			response_format: str = "mp3",
			speed: float = 1.0,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> SynthesisResult:
		assert_capability(model, "synthesis")
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		return self._run_rotation_generic(
			model,
			rid,
			sid,
			cfg,
			lambda key: self._synthesis_client.synthesize(
				api_key=key.raw_key,
				model=model,
				text=text,
				voice=voice,
				session_id=sid,
				api_key_id=key.key_id,
				response_format=response_format,
				speed=speed,
				request_id=rid,
				),
			capability="synthesis",
			)

	async def async_synthesize(
			self,
			text: str,
			voice: str,
			model: str = DEFAULT_SYNTHESIS_MODEL,
			response_format: str = "mp3",
			speed: float = 1.0,
			config: RequestConfig | None = None,
			session_id: str | None = None,
			) -> SynthesisResult:
		assert_capability(model, "synthesis")
		rid = str(uuid.uuid4())
		sid = self._resolve_session(session_id, rid)
		cfg = config or RequestConfig(retries=self._config.max_retries)
		return await self._async_run_rotation_generic(
			model,
			rid,
			sid,
			cfg,
			lambda key: self._synthesis_client.async_synthesize(
				api_key=key.raw_key,
				model=model,
				text=text,
				voice=voice,
				session_id=sid,
				api_key_id=key.key_id,
				response_format=response_format,
				speed=speed,
				request_id=rid,
				),
			capability="synthesis",
			)

	# ------------------------------------------------------------------
	# Introspection
	# ------------------------------------------------------------------

	def get_session_stats(self, session_id: str) -> dict | None:
		return self._session_manager.get_stats(session_id)

	def get_global_stats(self) -> dict:
		return self._usage_tracker.snapshot()

	def get_key_pool_status(self) -> list[dict]:
		return self._scheduler.status_summary()

	def health(self):
		return self._health_service.snapshot(
			key_status=self.get_key_pool_status(),
			active_sessions=self._session_manager.active_count(),
			global_stats=self.get_global_stats(),
			)

	def cleanup_sessions(self) -> int:
		return self._session_manager.expire_old_sessions()
