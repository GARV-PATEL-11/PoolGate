"""Shared fixtures for integration tests."""

from __future__ import annotations

import pytest

from poolgate.services.provider import GroqService


def _set_groq_keys(monkeypatch, keys: list[str]) -> None:
    monkeypatch.setenv("TOTAL_GROQ_KEYS", str(len(keys)))
    for i, key in enumerate(keys, start=1):
        monkeypatch.setenv(f"GROQ_API_KEY_{i:02d}", key)


@pytest.fixture
def two_key_service(monkeypatch) -> GroqService:
    _set_groq_keys(monkeypatch, ["gsk_integration_key_1", "gsk_integration_key_2"])
    return GroqService()


@pytest.fixture
def three_key_service(monkeypatch) -> GroqService:
    _set_groq_keys(
        monkeypatch,
        ["gsk_integration_key_1", "gsk_integration_key_2", "gsk_integration_key_3"],
    )
    return GroqService()
