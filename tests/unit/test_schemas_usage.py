"""Unit tests for schemas/usage.py — TokenUsage auto-fill and QuotaStatus validators."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from schemas.usage import QuotaStatus, TokenUsage


class TestTokenUsageAutoFill:

	def test_total_is_auto_filled_when_zero_and_parts_supplied(self):
		u = TokenUsage(prompt_tokens=10, completion_tokens=5)
		assert u.total_tokens == 15

	def test_total_is_preserved_when_explicitly_set(self):
		u = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=20)
		assert u.total_tokens == 20

	def test_total_stays_zero_when_parts_are_also_zero(self):
		u = TokenUsage()
		assert u.total_tokens == 0

	def test_only_prompt_fills_total(self):
		u = TokenUsage(prompt_tokens=7)
		assert u.total_tokens == 7

	def test_only_completion_fills_total(self):
		u = TokenUsage(completion_tokens=3)
		assert u.total_tokens == 3

	def test_negative_tokens_raise(self):
		with pytest.raises(ValidationError):
			TokenUsage(prompt_tokens=-1)


class TestTokenUsageAddition:

	def test_add_accumulates_all_fields(self):
		a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
		b = TokenUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6)
		c = a + b
		assert c.prompt_tokens == 14
		assert c.completion_tokens == 7
		assert c.total_tokens == 21

	def test_add_zero_preserves_values(self):
		a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
		b = TokenUsage()
		c = a + b
		assert c.prompt_tokens == 10
		assert c.total_tokens == 15

	def test_add_returns_new_instance(self):
		a = TokenUsage(prompt_tokens=5, completion_tokens=2, total_tokens=7)
		b = TokenUsage(prompt_tokens=3, completion_tokens=1, total_tokens=4)
		c = a + b
		assert a.prompt_tokens == 5
		assert b.prompt_tokens == 3
		assert c.prompt_tokens == 8


class TestQuotaStatus:

	def _make(self, **kwargs):
		now = datetime.now(timezone.utc)
		base = dict(api_key_id="key_0", window_start=now, window_end=now)
		base.update(kwargs)
		return QuotaStatus(**base)

	def test_remaining_computed_from_limit(self):
		qs = self._make(requests_used=30, requests_limit=100)
		assert qs.remaining_requests == 70

	def test_remaining_tokens_computed_from_limit(self):
		qs = self._make(tokens_used=5000, tokens_limit=10000)
		assert qs.remaining_tokens == 5000

	def test_remaining_clamps_to_zero_when_over_limit(self):
		qs = self._make(requests_used=120, requests_limit=100)
		assert qs.remaining_requests == 0

	def test_exhausted_when_requests_depleted(self):
		qs = self._make(requests_used=100, requests_limit=100)
		assert qs.exhausted is True

	def test_exhausted_when_tokens_depleted(self):
		qs = self._make(tokens_used=10000, tokens_limit=10000)
		assert qs.exhausted is True

	def test_not_exhausted_when_resources_remain(self):
		qs = self._make(requests_used=50, requests_limit=100, tokens_used=1000, tokens_limit=5000)
		assert qs.exhausted is False

	def test_no_limit_means_remaining_stays_none(self):
		qs = self._make(requests_used=10)
		assert qs.remaining_requests is None

	def test_exhausted_false_by_default_when_no_limits(self):
		qs = self._make()
		assert qs.exhausted is False
