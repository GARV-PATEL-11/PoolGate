"""
End-to-end: construct a real GroqService against a mocked Groq SDK and run
a full invoke() call through key acquisition, the live API call, tracking,
and session recording.

This is the cheapest test that would have caught the original critical bug
(GroqService() failing to construct because ChatClient was uninstantiable)
in any CI run — see scripts/smoke_test.py for an even smaller, faster
version of the same idea meant to run outside pytest entirely.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from services.provider_service import GroqService


def _mock_completion(text: str):
	completion = MagicMock()
	completion.choices = [MagicMock(message=MagicMock(content=text), finish_reason="stop")]
	completion.usage = MagicMock(prompt_tokens=3, completion_tokens=1, total_tokens=4)
	return completion


@pytest.fixture
def service(monkeypatch):
	monkeypatch.setenv("TOTAL_GROQ_KEYS", "3")
	monkeypatch.setenv("GROQ_API_KEY_01", "gsk_test_1")
	monkeypatch.setenv("GROQ_API_KEY_02", "gsk_test_2")
	monkeypatch.setenv("GROQ_API_KEY_03", "gsk_test_3")
	svc = GroqService()

	mock_sdk = MagicMock()
	mock_sdk.chat.completions.create.return_value = _mock_completion("42")
	monkeypatch.setattr(svc._chat_client, "_sync_sdk", lambda api_key: mock_sdk)
	return svc


def test_groq_service_constructs_successfully(monkeypatch):
	"""
	The single highest-value test in this suite: GroqService() must not
	raise. Before the fix, this failed unconditionally with TypeError
	because ChatClient (constructed in __init__) was an abstract class
	with unimplemented async_invoke/async_stream.
	"""
	monkeypatch.setenv("TOTAL_GROQ_KEYS", "1")
	monkeypatch.setenv("GROQ_API_KEY_01", "gsk_smoke_test_key")
	svc = GroqService()
	assert svc is not None


def test_invoke_full_lifecycle(service):
	response = service.invoke("What is six times seven?", model="llama-3.3-70b-versatile")
	assert response.text == "42"

	pool_status = service.get_key_pool_status()
	used_key = next((k for k in pool_status if k["requests_per_minute"] >= 1), None)
	assert used_key is not None

	stats = service.get_global_stats()
	assert stats["successful_requests"] >= 1


def test_chat_rejects_invalid_role(service):
	"""Regression test for H2: malformed roles must fail with a typed
	PoolGate exception, not be silently forwarded to the SDK."""
	from exceptions.request import InvalidMessageRoleError

	with pytest.raises(InvalidMessageRoleError):
		service.chat(messages=[{"role": "narrator", "content": "hi"}])


def test_health_reports_active_keys(service):
	health = service.health()
	assert health.active_keys >= 1
