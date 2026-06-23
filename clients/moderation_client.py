"""
moderation_client.py — ModerationClient

Safety / content classification using Groq's prompt-guard and safeguard models.

These models use the chat completions endpoint under the hood, but their
contract differs from conversational models:
  Input   → a single text string to classify
  Output  → ModerationResult (structured label + raw text + telemetry)

Supported models:
  meta-llama/llama-prompt-guard-2-22m
  meta-llama/llama-prompt-guard-2-86m
  openai/gpt-oss-safeguard-20b

──────────────────────────────────────────────
Prompt Guard labels  (llama-prompt-guard-*)
  SAFE       — benign user input
  JAILBREAK  — direct jailbreak / prompt injection attempt
  INDIRECT   — indirect injection (e.g. hidden in a retrieved document)

GPT-OSS-Safeguard labels  (openai/gpt-oss-safeguard-*)
  safe       — content is within policy
  unsafe     — content violates policy
──────────────────────────────────────────────

Public methods:
  moderate()   — blocking classification
  async_moderate()  — async classification
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from clients.base import (
    _choice_text,
    _first_choice,
    _new_rid,
    _parse_usage,
    BaseGroqClient,
)
from clients.capabilities import ModerationCapability
from schemas.runtime import RequestConfig, TokenUsage

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass
class ModerationResult:
    """Structured result from a moderation / classification call."""

    label: str  # first line of the model response: "SAFE", "JAILBREAK", "unsafe", …
    raw_text: str  # full model output — may contain a brief explanation after the label
    model: str
    usage: TokenUsage
    latency: float
    session_id: str
    request_id: str
    api_key_id: str


# ---------------------------------------------------------------------------
# Prompt format helpers
# ---------------------------------------------------------------------------

_SAFEGUARD_MODELS: frozenset[str] = frozenset({"openai/gpt-oss-safeguard-20b"})


def _prompt_guard_messages(text: str) -> list[dict[str, str]]:
    """
    Llama Prompt Guard is fine-tuned to classify the raw user turn directly.
    No system prompt — injecting one degrades accuracy.
    """
    return [{"role": "user", "content": text}]


def _safeguard_messages(text: str) -> list[dict[str, str]]:
    """
    GPT-OSS-Safeguard follows the OpenAI moderation-style prompt pattern.
    A concise system prompt focusing the model on label-only output.
    """
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


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ModerationClient(BaseGroqClient, ModerationCapability):
    """
    Stateless moderation client.

    Automatically selects the correct prompt format based on the model name so
    callers never need to know which format each model expects.

    Temperature is always forced to 0.0 — classification must be deterministic.
    max_tokens is capped at 16 unless the caller sets it lower in config, because
    moderation models only need to output a single label word.
    """

    def _build_messages(self, model: str, text: str) -> list[dict[str, str]]:
        if model in _SAFEGUARD_MODELS:
            return _safeguard_messages(text)
        return _prompt_guard_messages(text)

    def _extract_label(self, raw: str) -> str:
        """Extract the classification label — always the first non-empty line."""
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return raw.strip()

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

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
        """
        Blocking content classification.

        Returns a ModerationResult.  Check .label for the safety verdict.
        Inspect .raw_text if the model appended a brief rationale.
        """
        rid = _new_rid(request_id)
        client = self._sync_sdk(api_key)
        messages = self._build_messages(model, text)
        start = time.perf_counter()

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,  # always deterministic for classification
                max_tokens=min(config.max_tokens or 16, 16),
                timeout=config.timeout,
                stream=False,
            )
        except Exception as exc:
            self._handle_sdk_error(exc, rid, api_key_id)

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

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

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
        """
        Async content classification.

        Identical contract to moderate() — uses the native async Groq SDK.
        """
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
