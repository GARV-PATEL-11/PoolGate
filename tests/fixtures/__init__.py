"""
Shared pytest fixtures for PoolGate tests.

All fixtures:
- Use numbered Groq API keys (TOTAL_GROQ_KEYS + GROQ_API_KEY_0X)
- Avoid real network calls
- Ensure env is set BEFORE service instantiation
"""

from __future__ import annotations

import pytest

from services.persistence_service import PersistenceService
from services.provider_service import GroqService
from tests.mocks import mock_sync_sdk

# ============================================================================
# Helpers
# ============================================================================


def _set_groq_keys(monkeypatch, keys: list[str]) -> None:
    """Standardized Groq key setup for all tests."""
    monkeypatch.setenv("TOTAL_GROQ_KEYS", str(len(keys)))

    for i, key in enumerate(keys, start=1):
        monkeypatch.setenv(f"GROQ_API_KEY_{i:02d}", key)


# ============================================================================
# Service fixtures
# ============================================================================


@pytest.fixture
def two_key_service(monkeypatch) -> GroqService:
    """GroqService initialized with two dummy keys."""
    _set_groq_keys(monkeypatch, ["gsk_key_alpha", "gsk_key_beta"])
    return GroqService()


@pytest.fixture
def single_key_service(monkeypatch) -> GroqService:
    """GroqService initialized with a single dummy key."""
    _set_groq_keys(monkeypatch, ["gsk_only_key"])
    return GroqService()


# ============================================================================
# Mocked SDK service
# ============================================================================


@pytest.fixture
def mocked_service(monkeypatch, two_key_service: GroqService):
    """GroqService with mocked SDK response."""
    sdk = mock_sync_sdk("mocked answer")

    monkeypatch.setattr(
        two_key_service._chat_client,
        "_sync_sdk",
        lambda _k: sdk,
    )

    return two_key_service, sdk


# ============================================================================
# Persistence-backed service
# ============================================================================


@pytest.fixture
def persisted_service(monkeypatch, tmp_path) -> tuple[GroqService, PersistenceService]:
    """GroqService with JSON persistence enabled."""
    _set_groq_keys(monkeypatch, ["gsk_persist_key"])

    path = tmp_path / "stats.json"
    persistence = PersistenceService.json(path)

    svc = GroqService(persistence=persistence)

    sdk = mock_sync_sdk("persisted answer")
    monkeypatch.setattr(svc._chat_client, "_sync_sdk", lambda _k: sdk)

    return svc, persistence


# ============================================================================
# Constants
# ============================================================================

FAST_MODEL = "llama-3.3-70b-versatile"
SMALL_MODEL = "llama-3.2-1b-preview"


# ============================================================================
# Utilities
# ============================================================================


def make_messages(*user_turns: str) -> list[dict[str, str]]:
    """Build chat messages from user strings."""
    return [{"role": "user", "content": t} for t in user_turns]


def make_chat_history(*turns: tuple[str, str]) -> list[dict[str, str]]:
    """Build chat history from (role, content) pairs."""
    return [{"role": role, "content": content} for role, content in turns]
