"""Shared fixtures for provider-layer tests.

Provider tests use mocked SDKs — no real API calls.
"""

from __future__ import annotations

import pytest

from services.provider_service import GroqService


def _set_groq_keys(monkeypatch, keys: list[str]) -> None:
    monkeypatch.setenv("TOTAL_GROQ_KEYS", str(len(keys)))
    for i, key in enumerate(keys, start=1):
        monkeypatch.setenv(f"GROQ_API_KEY_{i:02d}", key)


@pytest.fixture
def service(monkeypatch) -> GroqService:
    _set_groq_keys(monkeypatch, ["gsk_provider_test_key_1", "gsk_provider_test_key_2"])
    return GroqService()
