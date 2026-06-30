"""Shared fixtures for capability (client) unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from poolgate.services.provider import GroqService


def _set_groq_keys(monkeypatch, keys: list[str]) -> None:
    monkeypatch.setenv("TOTAL_GROQ_KEYS", str(len(keys)))
    for i, key in enumerate(keys, start=1):
        monkeypatch.setenv(f"GROQ_API_KEY_{i:02d}", key)


@pytest.fixture
def two_key_service(monkeypatch) -> GroqService:
    _set_groq_keys(monkeypatch, ["gsk_e2e_key_1", "gsk_e2e_key_2"])
    return GroqService()


@pytest.fixture
def three_key_service(monkeypatch) -> GroqService:
    _set_groq_keys(monkeypatch, ["gsk_e2e_key_1", "gsk_e2e_key_2", "gsk_e2e_key_3"])
    return GroqService()


@pytest.fixture
def single_key_service(monkeypatch) -> GroqService:
    _set_groq_keys(monkeypatch, ["gsk_e2e_key_1"])
    return GroqService()


def _mock_stream_chunks(texts: list[str]) -> MagicMock:
    chunks = []
    for text in texts:
        chunk = MagicMock(x_groq=None)
        chunk.choices = [MagicMock(delta=MagicMock(content=text))]
        chunks.append(chunk)
    final = MagicMock()
    final.choices = [MagicMock(delta=MagicMock(content=""))]
    final.x_groq = MagicMock()
    final.x_groq.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chunks.append(final)

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=iter(chunks))
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx
