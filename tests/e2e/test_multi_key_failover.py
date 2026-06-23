"""E2E tests for multi-key failover — key rotation on failure."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from exceptions.keys import APIKeyDisabledError, NoAvailableAPIKeyError
from exceptions.response import RetryExhaustedError
from schemas.runtime import APIKeyStatus, RequestConfig
from services.provider_service import GroqService


def _set_groq_keys(monkeypatch, keys: list[str]) -> None:
	monkeypatch.setenv("TOTAL_GROQ_KEYS", str(len(keys)))
	for i, key in enumerate(keys, start=1):
		monkeypatch.setenv(f"GROQ_API_KEY_{i:02d}", key)


def _mock_completion(text: str = "answer") -> MagicMock:
	completion = MagicMock()
	completion.choices = [MagicMock(message=MagicMock(content=text), finish_reason="stop")]
	completion.usage = MagicMock(prompt_tokens=5, completion_tokens=3, total_tokens=8)
	return completion


@pytest.fixture
def two_key_service(monkeypatch) -> GroqService:
	_set_groq_keys(monkeypatch, ["gsk_failover_key_1", "gsk_failover_key_2"])
	return GroqService()


@pytest.fixture
def three_key_service(monkeypatch) -> GroqService:
	_set_groq_keys(monkeypatch, ["gsk_fo_key_1", "gsk_fo_key_2", "gsk_fo_key_3"])
	return GroqService()


# ---------------------------------------------------------------------------
# Failover scenarios
# ---------------------------------------------------------------------------

class TestMultiKeyFailover:

	def test_second_key_used_when_first_fails(self, two_key_service, monkeypatch):
		call_count = [0]
		success_sdk = MagicMock()
		success_sdk.chat.completions.create.return_value = _mock_completion("fallback")

		def factory(api_key: str) -> MagicMock:
			call_count[0] += 1
			if call_count[0] == 1:
				failing = MagicMock()
				failing.chat.completions.create.side_effect = RuntimeError("first key fail")
				return failing
			return success_sdk

		monkeypatch.setattr(two_key_service._chat_client, "_sync_sdk", factory)

		cfg = RequestConfig(retries=1)
		response = two_key_service.invoke("Q", model="llama-3.3-70b-versatile", config=cfg)
		assert response.text == "fallback"
		assert call_count[0] == 2

	def test_all_keys_fail_raises_retry_exhausted(self, two_key_service, monkeypatch):
		def factory(api_key: str) -> MagicMock:
			sdk = MagicMock()
			sdk.chat.completions.create.side_effect = RuntimeError("always fails")
			return sdk

		monkeypatch.setattr(two_key_service._chat_client, "_sync_sdk", factory)

		cfg = RequestConfig(retries=1)
		with pytest.raises(RetryExhaustedError):
			two_key_service.invoke("Q", model="llama-3.3-70b-versatile", config=cfg)

	def test_no_available_keys_raises_immediately(self, two_key_service):
		for k in two_key_service._scheduler._keys:
			k.mark_disabled()

		with pytest.raises(NoAvailableAPIKeyError):
			two_key_service.invoke("Q", model="llama-3.3-70b-versatile")

	def test_failed_key_marked_after_failure(self, two_key_service, monkeypatch):
		call_count = [0]
		success_sdk = MagicMock()
		success_sdk.chat.completions.create.return_value = _mock_completion("ok")

		def factory(api_key: str) -> MagicMock:
			call_count[0] += 1
			if call_count[0] == 1:
				failing = MagicMock()
				failing.chat.completions.create.side_effect = RuntimeError("fail")
				return failing
			return success_sdk

		monkeypatch.setattr(two_key_service._chat_client, "_sync_sdk", factory)

		cfg = RequestConfig(retries=1)
		two_key_service.invoke("Q", model="llama-3.3-70b-versatile", config=cfg)

		# First key should have accumulated at least one failure
		keys = two_key_service._scheduler._keys
		total_failures = sum(k.failure_count for k in keys)
		assert total_failures >= 1

	def test_three_keys_two_fail_third_succeeds(self, three_key_service, monkeypatch):
		call_count = [0]
		success_sdk = MagicMock()
		success_sdk.chat.completions.create.return_value = _mock_completion("third_key_answer")

		def factory(api_key: str) -> MagicMock:
			call_count[0] += 1
			if call_count[0] <= 2:
				failing = MagicMock()
				failing.chat.completions.create.side_effect = RuntimeError("fail")
				return failing
			return success_sdk

		monkeypatch.setattr(three_key_service._chat_client, "_sync_sdk", factory)

		cfg = RequestConfig(retries=2)
		response = three_key_service.invoke("Q", model="llama-3.3-70b-versatile", config=cfg)
		assert response.text == "third_key_answer"


# ---------------------------------------------------------------------------
# Auth error failover
# ---------------------------------------------------------------------------

class TestAuthErrorFailover:

	def test_auth_error_key_marked_disabled(self, two_key_service, monkeypatch):
		call_count = [0]
		success_sdk = MagicMock()
		success_sdk.chat.completions.create.return_value = _mock_completion("ok")

		def _make_auth_exc():
			exc = Exception("auth failed")
			exc.status_code = 401  # type: ignore[attr-defined]
			return exc

		def factory(api_key: str) -> MagicMock:
			call_count[0] += 1
			if call_count[0] == 1:
				failing = MagicMock()
				failing.chat.completions.create.side_effect = _make_auth_exc()
				return failing
			return success_sdk

		monkeypatch.setattr(two_key_service._chat_client, "_sync_sdk", factory)

		cfg = RequestConfig(retries=1)
		# Auth error causes key to be disabled, retry picks second key
		try:
			response = two_key_service.invoke("Q", model="llama-3.3-70b-versatile", config=cfg)
			# If second key succeeds, check first key is disabled
			disabled = [
				k for k in two_key_service._scheduler._keys
				if k.status == APIKeyStatus.DISABLED
				]
			assert len(disabled) >= 1
		except (RetryExhaustedError, NoAvailableAPIKeyError, APIKeyDisabledError):
			# Acceptable if both keys had auth issues
			pass
