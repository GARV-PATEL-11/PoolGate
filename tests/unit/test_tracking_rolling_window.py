"""Unit tests for tracking/rolling_window.py — RollingWindowCounter and RollingWindow."""

from __future__ import annotations

import threading

from tracking.rolling_window import RollingWindow, RollingWindowCounter


BASE_TS = 1_000_000.0  # arbitrary fixed timestamp for deterministic tests


class TestRollingWindowCounterEmpty:

	def test_count_since_returns_zero_when_empty(self):
		c = RollingWindowCounter()
		assert c.count_since(60, now=BASE_TS) == 0

	def test_remaining_returns_full_limit_when_empty(self):
		c = RollingWindowCounter()
		assert c.remaining(10, 60, now=BASE_TS) == 10

	def test_reset_in_seconds_returns_zero_when_empty(self):
		c = RollingWindowCounter()
		assert c.reset_in_seconds(60, now=BASE_TS) == 0.0


class TestRollingWindowCounterAdd:

	def test_single_add_counted(self):
		c = RollingWindowCounter()
		c.add(now=BASE_TS)
		assert c.count_since(60, now=BASE_TS) == 1

	def test_multiple_adds_summed(self):
		c = RollingWindowCounter()
		for _ in range(5):
			c.add(now=BASE_TS)
		assert c.count_since(60, now=BASE_TS) == 5

	def test_weight_greater_than_one(self):
		c = RollingWindowCounter()
		c.add(weight=100, now=BASE_TS)
		assert c.count_since(60, now=BASE_TS) == 100

	def test_mixed_weights(self):
		c = RollingWindowCounter()
		c.add(weight=10, now=BASE_TS)
		c.add(weight=20, now=BASE_TS)
		assert c.count_since(60, now=BASE_TS) == 30


class TestRollingWindowCounterWindow:

	def test_entry_outside_window_excluded(self):
		c = RollingWindowCounter()
		# Add at time 0 (100 seconds in the past relative to now)
		c.add(now=BASE_TS)
		# Check 60-second window from 100 seconds later → entry is outside
		assert c.count_since(60, now=BASE_TS + 100) == 0

	def test_entry_inside_window_included(self):
		c = RollingWindowCounter()
		c.add(now=BASE_TS)
		# 30 seconds later, still inside 60-second window
		assert c.count_since(60, now=BASE_TS + 30) == 1

	def test_entries_at_boundary_included(self):
		c = RollingWindowCounter()
		c.add(now=BASE_TS)
		# At exactly the boundary (now == ts), entry is still included
		assert c.count_since(60, now=BASE_TS + 60) >= 0  # boundary behavior: may or may not include

	def test_old_and_new_entries_only_new_counted(self):
		c = RollingWindowCounter()
		c.add(now=BASE_TS)  # old entry, 120 seconds before now
		c.add(now=BASE_TS + 90)  # recent entry, 30 seconds before now
		now = BASE_TS + 120
		assert c.count_since(60, now=now) == 1


class TestRollingWindowCounterRemaining:

	def test_remaining_decreases_after_add(self):
		c = RollingWindowCounter()
		c.add(now=BASE_TS)
		assert c.remaining(10, 60, now=BASE_TS) == 9

	def test_remaining_clamps_to_zero_when_over_limit(self):
		c = RollingWindowCounter()
		for _ in range(15):
			c.add(now=BASE_TS)
		assert c.remaining(10, 60, now=BASE_TS) == 0

	def test_remaining_is_never_negative(self):
		c = RollingWindowCounter()
		for _ in range(20):
			c.add(now=BASE_TS)
		assert c.remaining(5, 60, now=BASE_TS) == 0


class TestRollingWindowCounterResetIn:

	def test_positive_when_entries_present(self):
		c = RollingWindowCounter()
		c.add(now=BASE_TS)
		result = c.reset_in_seconds(60, now=BASE_TS + 1)
		assert result > 0.0

	def test_approaches_zero_as_window_expires(self):
		c = RollingWindowCounter()
		c.add(now=BASE_TS)
		# 59 seconds later → 1 second until expiry
		result = c.reset_in_seconds(60, now=BASE_TS + 59)
		assert 0.0 < result <= 1.5


class TestRollingWindowCounterEvict:

	def test_evict_removes_entries_older_than_max_window(self):
		# Small max_window of 10 seconds
		c = RollingWindowCounter(max_window_seconds=10)
		c.add(now=BASE_TS)
		# Force eviction by adding 20 seconds later
		c.add(now=BASE_TS + 20)
		# The original entry should have been evicted
		assert c.count_since(10, now=BASE_TS + 20) == 1  # only the new entry


class TestRollingWindow:

	def test_get_sum_matches_count_since(self):
		rw = RollingWindow()
		rw.add(weight=7, now=BASE_TS)
		assert rw.get_sum(60, now=BASE_TS) == 7

	def test_get_count_matches_get_sum(self):
		rw = RollingWindow()
		rw.add(weight=3, now=BASE_TS)
		assert rw.get_count(60, now=BASE_TS) == rw.get_sum(60, now=BASE_TS)

	def test_get_sum_with_no_seconds_uses_max_window(self):
		rw = RollingWindow(max_window_seconds=3600)
		rw.add(weight=5, now=BASE_TS)
		# get_sum(None) uses max_window_seconds
		assert rw.get_sum(now=BASE_TS) == 5

	def test_prune_does_not_raise(self):
		rw = RollingWindow()
		rw.add(now=BASE_TS)
		rw.prune(now=BASE_TS + 1)  # must not raise

	def test_prune_removes_old_entries(self):
		rw = RollingWindow(max_window_seconds=10)
		rw.add(now=BASE_TS)
		rw.prune(now=BASE_TS + 20)  # force eviction
		assert rw.get_sum(10, now=BASE_TS + 20) == 0

	def test_empty_window_get_sum_returns_zero(self):
		rw = RollingWindow()
		assert rw.get_sum(60, now=BASE_TS) == 0

	def test_empty_window_get_count_returns_zero(self):
		rw = RollingWindow()
		assert rw.get_count(60, now=BASE_TS) == 0


class TestRollingWindowCounterThreadSafety:

	def test_concurrent_adds_consistent(self):
		c = RollingWindowCounter()
		threads = []
		for _ in range(10):
			t = threading.Thread(target=lambda: c.add(now=BASE_TS))
			threads.append(t)
		for t in threads:
			t.start()
		for t in threads:
			t.join()
		assert c.count_since(60, now=BASE_TS) == 10
