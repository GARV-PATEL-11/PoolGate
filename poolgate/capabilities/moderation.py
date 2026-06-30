"""ModerationCapability — safety/content classification."""

from __future__ import annotations

import time
from dataclasses import dataclass

from poolgate.providers.base import (
    _choice_text,
    _first_choice,
    _new_rid,
    _parse_usage,
)
from poolgate.providers.groq.capabilities import ModerationCapability as _ModerationABC
from poolgate.providers.groq.client import GroqProvider
from poolgate.schemas.common.runtime import RequestConfig, TokenUsage


@dataclass
class ModerationResult:
    """Structured result from a moderation/classification call."""

    label: str
    raw_text: str
    model: str
    usage: TokenUsage
    latency: float
    session_id: str
    request_id: str
    api_key_id: str


_SAFEGUARD_MODELS: frozenset[str] = frozenset({"openai/gpt-oss-safeguard-20b"})


def _prompt_guard_messages(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def _safeguard_messages(text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a content moderation assistant. "
                "Classify the following text as exactly 'safe' or 'unsafe'. "
                "Output only the label — no explanation."
            ),
        },
        {"role": "user", "content": text},
    ]


class ModerationCapability(GroqProvider, _ModerationABC):
    """Stateless moderation client."""

    def _build_messages(self, model: str, text: str) -> list[dict[str, str]]:
        if model in _SAFEGUARD_MODELS:
            return _safeguard_messages(text)
        return _prompt_guard_messages(text)

    def _extract_label(self, raw: str) -> str:
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return raw.strip()

    def moderate(
        self,
        api_key: str,
        model: str,
        text: str,
        config: RequestConfig,
        session_id: str,
        api_key_id: str,
        request_id: str | None = None,
    ) -> ModerationResult:
        rid = _new_rid(request_id)
        client = self._sync_sdk(api_key)
        messages = self._build_messages(model, text)
        start = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,
                max_tokens=min(config.max_tokens or 16, 16),
                timeout=config.timeout,
                stream=False,
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise
        raw_text = _choice_text(_first_choice(completion, rid), rid)
        return ModerationResult(
            label=self._extract_label(raw_text),
            raw_text=raw_text,
            model=model,
            usage=_parse_usage(completion),
            latency=time.perf_counter() - start,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
        )

    async def async_moderate(
        self,
        api_key: str,
        model: str,
        text: str,
        config: RequestConfig,
        session_id: str,
        api_key_id: str,
        request_id: str | None = None,
    ) -> ModerationResult:
        rid = _new_rid(request_id)
        client = self._async_sdk(api_key)
        messages = self._build_messages(model, text)
        start = time.perf_counter()
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,
                max_tokens=min(config.max_tokens or 16, 16),
                timeout=config.timeout,
                stream=False,
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)
            raise
        raw_text = _choice_text(_first_choice(completion, rid), rid)
        return ModerationResult(
            label=self._extract_label(raw_text),
            raw_text=raw_text,
            model=model,
            usage=_parse_usage(completion),
            latency=time.perf_counter() - start,
            session_id=session_id,
            request_id=rid,
            api_key_id=api_key_id,
        )
