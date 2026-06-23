"""E2E tests for rate limit handling — key cooldown and rotation on 429."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from exceptions.keys import NoAvailableAPIKeyError
from exceptions.rate_limit import RateLimitExceededError
from exceptions.response import RetryExhaustedError
from schemas.runtime import APIKeyStatus, RequestConfig
from services.provider_service import GroqService


def _set_groq_keys(monkeypatch, keys: list[str]) -> None:
    monkeypatch.setenv("TOTAL_GROQ_KEYS", str(len(keys)))
    for i, key in enumerate(keys, start=1):
        monkeypatch.setenv(f"GROQ_API_KEY_{i:02d}", key)


def _mock_completion(text: str = "ok") -> MagicMock:
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=text), finish_reason="stop")]
    completion.usage = MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    return completion


class RateLimitError(Exception):
    pass


@pytest.fixture
def two_key_service(monkeypatch) -> GroqService:
    _set_groq_keys(monkeypatch, ["gsk_rl_key_1", "gsk_rl_key_2"])
    return GroqService()


@pytest.fixture
def single_key_service(monkeypatch) -> GroqService:
    _set_groq_keys(monkeypatch, ["gsk_rl_only_key"])
    return GroqService()


# ---------------------------------------------------------------------------
# Rate limit triggers key rotation
# ---------------------------------------------------------------------------


class TestRateLimitKeyRotation:

    def test_rate_limited_key_triggers_rotation_to_second(self, two_key_service, monkeypatch):
        call_count = [0]
        success_sdk = MagicMock()
        success_sdk.chat.completions.create.return_value = _mock_completion("from_key_2")

        def factory(api_key: str) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                failing = MagicMock()
                exc = RateLimitError("rate limited")
                exc.response = None  # type: ignore[attr-defined]
                failing.chat.completions.create.side_effect = exc
                return failing
            return success_sdk

        monkeypatch.setattr(two_key_service._chat_client, "_sync_sdk", factory)

        cfg = RequestConfig(retries=1)
        response = two_key_service.invoke("Q", model="llama-3.3-70b-versatile", config=cfg)
        assert response.text == "from_key_2"

    def test_rate_limited_key_enters_cooldown_state(self, two_key_service, monkeypatch):
        call_count = [0]
        success_sdk = MagicMock()
        success_sdk.chat.completions.create.return_value = _mock_completion("ok")

        def factory(api_key: str) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                failing = MagicMock()
                exc = RateLimitError("rl")
                exc.response = None  # type: ignore[attr-defined]
                failing.chat.completions.create.side_effect = exc
                return failing
            return success_sdk

        monkeypatch.setattr(two_key_service._chat_client, "_sync_sdk", factory)

        cfg = RequestConfig(retries=1)
        two_key_service.invoke("Q", model="llama-3.3-70b-versatile", config=cfg)

        # One key should have been rate limited
        rate_limited = [
            k
            for k in two_key_service._scheduler._keys
            if k.status == APIKeyStatus.RATE_LIMITED or k.consecutive_429s > 0
        ]
        assert len(rate_limited) >= 1

    def test_consecutive_failures_increment_on_rate_limit(self, two_key_service, monkeypatch):
        success_sdk = MagicMock()
        success_sdk.chat.completions.create.return_value = _mock_completion("ok")
        call_count = [0]

        def factory(api_key: str) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                failing = MagicMock()
                exc = RateLimitError("rl")
                exc.response = None  # type: ignore[attr-defined]
                failing.chat.completions.create.side_effect = exc
                return failing
            return success_sdk

        monkeypatch.setattr(two_key_service._chat_client, "_sync_sdk", factory)

        cfg = RequestConfig(retries=1)
        two_key_service.invoke("Q", model="llama-3.3-70b-versatile", config=cfg)

        total_429s = sum(k.consecutive_429s for k in two_key_service._scheduler._keys)
        assert total_429s >= 1


# ---------------------------------------------------------------------------
# All keys rate limited
# ---------------------------------------------------------------------------


class TestAllKeysRateLimited:

    def test_single_key_rate_limited_raises(self, single_key_service, monkeypatch):
        def factory(api_key: str) -> MagicMock:
            sdk = MagicMock()
            exc = RateLimitError("rl")
            exc.response = None  # type: ignore[attr-defined]
            sdk.chat.completions.create.side_effect = exc
            return sdk

        monkeypatch.setattr(single_key_service._chat_client, "_sync_sdk", factory)

        cfg = RequestConfig(retries=0)
        with pytest.raises((RetryExhaustedError, RateLimitExceededError)):
            single_key_service.invoke("Q", model="llama-3.3-70b-versatile", config=cfg)

    def test_all_keys_rate_limited_before_invoke_raises_no_available_key(
        self,
        two_key_service,
    ):
        # Mark both keys as rate limited with long cooldown
        for k in two_key_service._scheduler._keys:
            k.record_request_start()
            k.record_failure(is_rate_limit=True, cooldown_secs=3600.0)

        with pytest.raises(NoAvailableAPIKeyError):
            two_key_service.invoke("Q", model="llama-3.3-70b-versatile")
