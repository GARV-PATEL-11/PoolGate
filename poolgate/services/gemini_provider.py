"""GeminiService — PoolGate facade for Google Gemini key pooling."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from poolgate.capabilities.gemini.chat import GeminiChatCapability
from poolgate.capabilities.gemini.structured import GeminiStructuredCapability
from poolgate.capabilities.gemini.tools import GeminiToolCapability
from poolgate.core.config import GroqConfig
from poolgate.core.gemini_config import GeminiConfig
from poolgate.core.logger import LoggerManager, RequestContext
from poolgate.exceptions.configuration import EmptyKeyPoolError
from poolgate.exceptions.keys import APIKeyDisabledError, NoAvailableAPIKeyError
from poolgate.exceptions.output import SessionExpiredError, StructuredOutputError
from poolgate.exceptions.request import MissingPromptError
from poolgate.exceptions.response import RetryExhaustedError
from poolgate.persistence.session import SessionManager
from poolgate.persistence.snapshots import DailySnapshotRepository as PersistenceService
from poolgate.persistence.snapshots import RequestJournal
from poolgate.pool.key_pool import APIKeyState
from poolgate.pool.scheduler import RequestScheduler
from poolgate.providers.gemini.models import assert_gemini_capability
from poolgate.schemas.common.runtime import (
    BatchResult,
    BatchSummary,
    GroqResponse,
    RequestConfig,
    RuntimeChatMessage,
    TokenUsage,
)
from poolgate.services.health import HealthService
from poolgate.services.retry import is_auth_error, is_rate_limit
from poolgate.tracking.manager import TrackingManager

T = TypeVar("T", bound=BaseModel)
_R = TypeVar("_R")

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _prompt_to_messages(prompt: str, system: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _validate_messages(messages: list[dict[str, str]]) -> None:
    for message in messages:
        RuntimeChatMessage(role=message.get("role", ""), content=message.get("content", ""))


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines[1:] if line.strip() != "```").strip()
    return text


def _sanitize_parse_error(exc: Exception, *, max_len: int = 200) -> str:
    message = str(exc).replace("\n", " ").replace("\r", " ").strip()
    if len(message) > max_len:
        message = message[:max_len].rstrip() + "…"
    return message


class GeminiService:
    """Production-facing Google Gemini facade with key scheduling and session tracking."""

    def __init__(
        self,
        config: GeminiConfig | None = None,
        session_id: str | None = None,
        debug_mode: bool = False,
        persistence: PersistenceService | None = None,
    ) -> None:
        self._config = config or GeminiConfig.from_env()
        self._debug = debug_mode or self._config.debug_mode

        self._log_manager = LoggerManager(
            self._config.paths,
            level=self._config.log_level,
            debug=self._debug,
        )
        self._logger = self._log_manager.get("gemini_service")

        self._session_manager = SessionManager(self._config.session_ttl_hours)
        self._tracking = TrackingManager()
        self._usage_tracker = self._tracking.usage_tracker
        self._health_service = HealthService()

        paths = self._config.paths

        if persistence is not None:
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
            self._log_manager.log_storage(event="load", tracker="all", path=paths.tracking_dir or "")
            self._logger.info("GeminiService loaded tracker history from persistence backend.")

        self._journal: RequestJournal | None = RequestJournal(paths.requests_dir) if paths.requests_dir else None

        keys = [
            APIKeyState.from_key(
                key_id=f"gemini_key_{index}",
                raw_key=key,
                max_parallel=self._config.max_active_requests,
            )
            for index, key in enumerate(self._config.api_keys)
        ]
        if not keys:
            raise EmptyKeyPoolError("GeminiService requires at least one API key.")

        # Cast to satisfy RequestScheduler's GroqConfig type annotation;
        # GeminiConfig has the same field layout so duck typing is safe at runtime.
        self._scheduler = RequestScheduler(keys, cast(GroqConfig, self._config), self._logger)
        self._chat_client = GeminiChatCapability()
        self._structured_client = GeminiStructuredCapability()
        self._tool_client = GeminiToolCapability()
        self._default_session_id = session_id

        self._logger.info(f"GeminiService initialized with {len(keys)} API key(s). debug={self._debug}")

    # ── Persistence ──────────────────────────────────────────────────────────

    def flush_tracking(self) -> None:
        """Persist current tracker state to JSON files. No-op if no persistence backend."""
        paths = self._config.paths
        if self._usage_persistence is not None:
            self._usage_persistence.flush_tracker(self._tracking.usage_tracker)
            self._log_manager.log_storage(event="flush", tracker="usage", path=paths.usage_json or "")
        if self._token_persistence is not None:
            self._token_persistence.flush_tracker(self._tracking.token_tracker)
            self._log_manager.log_storage(event="flush", tracker="tokens", path=paths.tokens_json or "")
        if self._account_persistence is not None:
            self._account_persistence.flush_tracker(self._tracking.account_tracker)
            self._log_manager.log_storage(event="flush", tracker="account", path=paths.account_json or "")

    # ── Session helpers ───────────────────────────────────────────────────────

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
        self._journal.append(
            {
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
            }
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

    # ── Key-rotation core (sync) ──────────────────────────────────────────────

    def _run_with_rotation(
        self,
        model: str,
        request_id: str,
        session_id: str,
        config: RequestConfig,
        call: Callable[[APIKeyState], GroqResponse],
        capability: str = "chat",
    ) -> GroqResponse:
        ctx = RequestContext(request_id=request_id, session_id=session_id, model=model)
        attempts = config.retries + 1
        last_exc: Exception | None = None

        self._log_manager.log_request(rid=request_id, sid=session_id, model=model, capability=capability, attempt=1)

        for attempt in range(attempts):
            key: APIKeyState | None = None
            start = time.perf_counter()
            try:
                key = self._scheduler.acquire_key(request_id, model=model)
                ctx.api_key_id = key.key_id
                ctx.retry_count = attempt
                self._log_manager.log_trace(
                    rid=request_id, stage="key_acquired", key_id=key.key_id, attempt=attempt + 1
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
                    rid=request_id,
                    model=model,
                    tokens_in=response.usage.prompt_tokens,
                    tokens_out=response.usage.completion_tokens,
                    latency=latency,
                    success=True,
                    finish_reason=(response.finish_reason.value if response.finish_reason else ""),
                )
                self._log_manager.log_performance(
                    rid=request_id,
                    model=model,
                    latency=latency,
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
                is_auth = is_auth_error(exc)
                if key:
                    if is_auth:
                        self._scheduler.mark_key_disabled(key)
                    else:
                        self._scheduler.mark_key_failure(key, is_rate_limit=is_rate_limit(exc))
                self._logger.warning(f"Attempt {attempt + 1}/{attempts} failed: {type(exc).__name__}: {exc}", ctx)
                self._log_manager.log_trace(
                    rid=request_id, stage="retry", attempt=attempt + 1, error=type(exc).__name__
                )
                if is_auth:
                    break

        self._log_manager.log_response(
            rid=request_id,
            model=model,
            tokens_in=0,
            tokens_out=0,
            latency=0.0,
            success=False,
            error=type(last_exc).__name__ if last_exc else "unknown",
        )
        self._log_manager.log_trace(rid=request_id, stage="exhausted", attempts=attempts)
        self._record_failure(session_id, model, retried=attempts > 1)
        raise RetryExhaustedError(
            f"All {attempts} attempt(s) failed. Last error: {last_exc}",
            attempts=attempts,
            last_exc=last_exc,
            request_id=request_id,
        ) from last_exc

    # ── Key-rotation core (async) ─────────────────────────────────────────────

    async def _async_run_with_rotation(
        self,
        model: str,
        request_id: str,
        session_id: str,
        config: RequestConfig,
        call: Callable[[APIKeyState], Awaitable[GroqResponse]],
        capability: str = "chat",
    ) -> GroqResponse:
        ctx = RequestContext(request_id=request_id, session_id=session_id, model=model)
        attempts = config.retries + 1
        last_exc: Exception | None = None

        self._log_manager.log_request(rid=request_id, sid=session_id, model=model, capability=capability, attempt=1)

        for attempt in range(attempts):
            key: APIKeyState | None = None
            start = time.perf_counter()
            try:
                key = await self._scheduler.async_acquire_key(request_id, model=model)
                ctx.api_key_id = key.key_id
                ctx.retry_count = attempt
                self._log_manager.log_trace(
                    rid=request_id, stage="key_acquired", key_id=key.key_id, attempt=attempt + 1
                )
                response: GroqResponse = await call(key)
                latency = time.perf_counter() - start
                self._scheduler.release_key(
                    key,
                    latency=latency,
                    tokens_in=response.usage.prompt_tokens,
                    tokens_out=response.usage.completion_tokens,
                )
                self._log_manager.log_response(
                    rid=request_id,
                    model=model,
                    tokens_in=response.usage.prompt_tokens,
                    tokens_out=response.usage.completion_tokens,
                    latency=latency,
                    success=True,
                    finish_reason=(response.finish_reason.value if response.finish_reason else ""),
                )
                self._log_manager.log_performance(
                    rid=request_id,
                    model=model,
                    latency=latency,
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
                is_auth = is_auth_error(exc)
                if key:
                    if is_auth:
                        self._scheduler.mark_key_disabled(key)
                    else:
                        self._scheduler.mark_key_failure(key, is_rate_limit=is_rate_limit(exc))
                self._logger.warning(f"Async attempt {attempt + 1}/{attempts} failed: {type(exc).__name__}: {exc}", ctx)
                self._log_manager.log_trace(
                    rid=request_id, stage="retry", attempt=attempt + 1, error=type(exc).__name__
                )
                if is_auth:
                    break

        self._log_manager.log_response(
            rid=request_id,
            model=model,
            tokens_in=0,
            tokens_out=0,
            latency=0.0,
            success=False,
            error=type(last_exc).__name__ if last_exc else "unknown",
        )
        self._log_manager.log_trace(rid=request_id, stage="exhausted", attempts=attempts)
        self._record_failure(session_id, model, retried=attempts > 1)
        raise RetryExhaustedError(
            f"All {attempts} async attempt(s) failed. Last error: {last_exc}",
            attempts=attempts,
            last_exc=last_exc,
            request_id=request_id,
        ) from last_exc

    # ── Chat ─────────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = DEFAULT_GEMINI_MODEL,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        stop: list[str] | str | None = None,
        timeout: float = 30.0,
        retries: int | None = None,
        session_id: str | None = None,
    ) -> GroqResponse:
        assert_gemini_capability(model, "chat")
        _validate_messages(messages)
        rid = str(uuid.uuid4())
        sid = self._resolve_session(session_id, rid)
        cfg = RequestConfig(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            timeout=timeout,
            retries=retries if retries is not None else self._config.max_retries,
        )
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

    async def async_chat(
        self,
        messages: list[dict[str, str]],
        model: str = DEFAULT_GEMINI_MODEL,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        stop: list[str] | str | None = None,
        timeout: float = 30.0,
        retries: int | None = None,
        session_id: str | None = None,
    ) -> GroqResponse:
        assert_gemini_capability(model, "chat")
        _validate_messages(messages)
        rid = str(uuid.uuid4())
        sid = self._resolve_session(session_id, rid)
        cfg = RequestConfig(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
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

    def invoke(
        self,
        prompt: str,
        model: str = DEFAULT_GEMINI_MODEL,
        system: str | None = None,
        config: RequestConfig | None = None,
        session_id: str | None = None,
    ) -> GroqResponse:
        if not prompt:
            raise MissingPromptError("prompt must be a non-empty string.")
        assert_gemini_capability(model, "chat")
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
        model: str = DEFAULT_GEMINI_MODEL,
        system: str | None = None,
        config: RequestConfig | None = None,
        session_id: str | None = None,
    ) -> GroqResponse:
        if not prompt:
            raise MissingPromptError("prompt must be a non-empty string.")
        assert_gemini_capability(model, "chat")
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

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        session_id: str | None = None,
        retries: int | None = None,
    ) -> Generator[str, None, None]:
        if messages is None:
            if not prompt:
                raise MissingPromptError("Either prompt or messages must be provided for streaming.")
            messages = _prompt_to_messages(prompt, system)
        assert_gemini_capability(model, "chat")
        _validate_messages(messages)
        rid = str(uuid.uuid4())
        sid = self._resolve_session(session_id, rid)
        cfg = RequestConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries if retries is not None else self._config.max_retries,
        )
        key = self._scheduler.acquire_key(rid, model=model)
        start = time.perf_counter()
        try:
            collected_usage: list[TokenUsage] = []
            yield from self._chat_client.stream(
                api_key=key.raw_key,
                model=model,
                messages=messages,
                config=cfg,
                session_id=sid,
                api_key_id=key.key_id,
                request_id=rid,
                on_usage=collected_usage.append,
            )
        except Exception as exc:
            is_auth = is_auth_error(exc)
            if is_auth:
                self._scheduler.mark_key_disabled(key)
            else:
                self._scheduler.mark_key_failure(key, is_rate_limit=is_rate_limit(exc))
            raise
        finally:
            latency = time.perf_counter() - start
            usage = collected_usage[-1] if collected_usage else TokenUsage()
            self._scheduler.release_key(
                key, latency=latency, tokens_in=usage.prompt_tokens, tokens_out=usage.completion_tokens
            )

    async def async_stream(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        system: str | None = None,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        session_id: str | None = None,
        retries: int | None = None,
    ) -> AsyncGenerator[str, None]:
        if messages is None:
            if not prompt:
                raise MissingPromptError("Either prompt or messages must be provided for streaming.")
            messages = _prompt_to_messages(prompt, system)
        assert_gemini_capability(model, "chat")
        _validate_messages(messages)
        rid = str(uuid.uuid4())
        sid = self._resolve_session(session_id, rid)
        cfg = RequestConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries if retries is not None else self._config.max_retries,
        )
        key = await self._scheduler.async_acquire_key(rid, model=model)
        start = time.perf_counter()
        try:
            collected_usage: list[TokenUsage] = []
            async for chunk in self._chat_client.async_stream(
                api_key=key.raw_key,
                model=model,
                messages=messages,
                config=cfg,
                session_id=sid,
                api_key_id=key.key_id,
                request_id=rid,
                on_usage=collected_usage.append,
            ):
                yield chunk
        except Exception as exc:
            is_auth = is_auth_error(exc)
            if is_auth:
                self._scheduler.mark_key_disabled(key)
            else:
                self._scheduler.mark_key_failure(key, is_rate_limit=is_rate_limit(exc))
            raise
        finally:
            latency = time.perf_counter() - start
            usage = collected_usage[-1] if collected_usage else TokenUsage()
            self._scheduler.release_key(
                key, latency=latency, tokens_in=usage.prompt_tokens, tokens_out=usage.completion_tokens
            )

    # ── Structured output ─────────────────────────────────────────────────────

    def structured(
        self,
        prompt: str,
        schema: type[T],
        model: str = DEFAULT_GEMINI_MODEL,
        system: str | None = None,
        config: RequestConfig | None = None,
        session_id: str | None = None,
        json_retries: int = 2,
    ) -> T:
        assert_gemini_capability(model, "structured")
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
                lambda key: self._structured_client.invoke_structured(
                    api_key=key.raw_key,
                    model=model,
                    messages=messages,
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
                    {"role": "user", "content": f"Fix the JSON only. Parse error: {_sanitize_parse_error(exc)}"},
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
        model: str = DEFAULT_GEMINI_MODEL,
        system: str | None = None,
        config: RequestConfig | None = None,
        session_id: str | None = None,
        json_retries: int = 2,
    ) -> T:
        assert_gemini_capability(model, "structured")
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
                lambda key: self._structured_client.async_invoke_structured(
                    api_key=key.raw_key,
                    model=model,
                    messages=messages,
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
                    {"role": "user", "content": f"Fix the JSON only. Parse error: {_sanitize_parse_error(exc)}"},
                ]

        raise StructuredOutputError(
            f"Failed to parse async structured output after {json_retries + 1} attempts: {last_exc}",
            raw_response=response.text if response else None,
            request_id=rid,
        )

    # ── Tool calling ──────────────────────────────────────────────────────────

    def invoke_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str = DEFAULT_GEMINI_MODEL,
        config: RequestConfig | None = None,
        session_id: str | None = None,
    ) -> GroqResponse:
        assert_gemini_capability(model, "tools")
        rid = str(uuid.uuid4())
        sid = self._resolve_session(session_id, rid)
        cfg = config or RequestConfig(retries=self._config.max_retries)
        return self._run_with_rotation(
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
                request_id=rid,
            ),
            capability="tools",
        )

    async def async_invoke_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str = DEFAULT_GEMINI_MODEL,
        config: RequestConfig | None = None,
        session_id: str | None = None,
    ) -> GroqResponse:
        assert_gemini_capability(model, "tools")
        rid = str(uuid.uuid4())
        sid = self._resolve_session(session_id, rid)
        cfg = config or RequestConfig(retries=self._config.max_retries)
        return await self._async_run_with_rotation(
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
                request_id=rid,
            ),
            capability="tools",
        )

    # ── Batch ─────────────────────────────────────────────────────────────────

    def batch_chat(
        self,
        prompts: list[str],
        model: str = DEFAULT_GEMINI_MODEL,
        system: str | None = None,
        config: RequestConfig | None = None,
        session_id: str | None = None,
    ) -> BatchSummary:
        results: list[BatchResult] = []
        total_tokens = TokenUsage()
        total_latency: float = 0.0
        cfg = config or RequestConfig(retries=self._config.max_retries)

        for i, prompt in enumerate(prompts):
            try:
                response = self.invoke(prompt, model=model, system=system, config=cfg, session_id=session_id)
                results.append(BatchResult(index=i, response=response, success=True))
                total_tokens.prompt_tokens += response.usage.prompt_tokens
                total_tokens.completion_tokens += response.usage.completion_tokens
                total_tokens.total_tokens += response.usage.total_tokens
                total_latency += response.latency
            except Exception as exc:
                results.append(BatchResult(index=i, error=str(exc), success=False))

        succeeded = sum(1 for r in results if r.success)
        return BatchSummary(
            total=len(prompts),
            succeeded=succeeded,
            failed=len(prompts) - succeeded,
            results=results,
            total_tokens=total_tokens,
            total_latency=total_latency,
        )

    async def async_batch_chat(
        self,
        prompts: list[str],
        model: str = DEFAULT_GEMINI_MODEL,
        system: str | None = None,
        config: RequestConfig | None = None,
        session_id: str | None = None,
        concurrency: int | None = None,
    ) -> BatchSummary:
        max_concurrent = concurrency or self._config.default_concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        cfg = config or RequestConfig(retries=self._config.max_retries)

        async def _bounded(i: int, prompt: str) -> BatchResult:
            async with semaphore:
                try:
                    response = await self.async_invoke(
                        prompt, model=model, system=system, config=cfg, session_id=session_id
                    )
                    return BatchResult(index=i, response=response, success=True)
                except Exception as exc:
                    return BatchResult(index=i, error=str(exc), success=False)

        results = await asyncio.gather(*(_bounded(i, p) for i, p in enumerate(prompts)))
        sorted_results = sorted(results, key=lambda r: r.index)
        total_tokens = TokenUsage()
        total_latency: float = 0.0
        for r in sorted_results:
            if r.success and r.response is not None:
                total_tokens.prompt_tokens += r.response.usage.prompt_tokens
                total_tokens.completion_tokens += r.response.usage.completion_tokens
                total_tokens.total_tokens += r.response.usage.total_tokens
                total_latency += r.response.latency

        succeeded = sum(1 for r in sorted_results if r.success)
        return BatchSummary(
            total=len(prompts),
            succeeded=succeeded,
            failed=len(prompts) - succeeded,
            results=sorted_results,
            total_tokens=total_tokens,
            total_latency=total_latency,
        )

    # ── Observability ─────────────────────────────────────────────────────────

    def get_key_pool_status(self) -> list[dict[str, Any]]:
        return self._scheduler.status_summary()

    def get_global_stats(self) -> dict[str, Any]:
        return self._tracking.snapshot()

    def health(self) -> Any:
        return self._health_service.snapshot(
            key_status=self.get_key_pool_status(),
            active_sessions=self._session_manager.active_count(),
            global_stats=self._tracking.usage_tracker.snapshot(),
        )
