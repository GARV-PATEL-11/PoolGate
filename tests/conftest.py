"""Shared pytest fixtures for the PoolGate test suite."""

from __future__ import annotations

import pytest

from core.config import GroqConfig
from key_manager.key_pool import APIKeyState, KeyPool


@pytest.fixture
def api_key_state() -> APIKeyState:
	return APIKeyState.from_key(key_id="key_0", raw_key="gsk_test_key", max_parallel=10)


@pytest.fixture
def key_pool() -> KeyPool:
	return KeyPool(["gsk_key_a", "gsk_key_b", "gsk_key_c"], max_parallel=10)


@pytest.fixture
def groq_config(monkeypatch) -> GroqConfig:
	monkeypatch.setenv("TOTAL_GROQ_KEYS", "2")
	monkeypatch.setenv("GROQ_API_KEY_01", "gsk_test_1")
	monkeypatch.setenv("GROQ_API_KEY_02", "gsk_test_2")

	return GroqConfig.from_env()


@pytest.fixture
def three_keys() -> list[APIKeyState]:
	return [APIKeyState.from_key(key_id=f"key_{i}", raw_key=f"gsk_{i}") for i in range(3)]
