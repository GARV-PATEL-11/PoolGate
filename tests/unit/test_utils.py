"""Unit tests for utils.py — SlidingWindowCounter, LatencyTracker, helpers."""

from __future__ import annotations

import time

import pytest

from utils import LatencyTracker, SlidingWindowCounter, clamp, now_ts, utc_now


class TestSlidingWindowCounter:
    def test_empty_counter_returns_zero(self):
        counter = SlidingWindowCounter(window_seconds=60)
        assert counter.count() == 0

    def test_recorded_events_are_counted(self):
        counter = SlidingWindowCounter(window_seconds=60)
        counter.record()
        counter.record()
        assert counter.count() == 2

    def test_old_events_are_evicted_outside_window(self):
        counter = SlidingWindowCounter(window_seconds=0.05)
        counter.record()
        assert counter.count() == 1
        time.sleep(0.1)
        assert counter.count() == 0

    def test_events_within_window_are_kept(self):
        counter = SlidingWindowCounter(window_seconds=60)
        for _ in range(5):
            counter.record()
        assert counter.count() == 5

    def test_count_is_stable_without_new_records(self):
        counter = SlidingWindowCounter(window_seconds=60)
        counter.record()
        assert counter.count() == 1
        assert counter.count() == 1


class TestLatencyTracker:
    def test_empty_tracker_returns_zero_average(self):
        tracker = LatencyTracker()
        assert tracker.average() == 0.0

    def test_empty_tracker_returns_zero_p95(self):
        tracker = LatencyTracker()
        assert tracker.p95() == 0.0

    def test_average_of_single_sample(self):
        tracker = LatencyTracker()
        tracker.record(0.5)
        assert tracker.average() == pytest.approx(0.5)

    def test_average_of_multiple_samples(self):
        tracker = LatencyTracker()
        tracker.record(1.0)
        tracker.record(3.0)
        assert tracker.average() == pytest.approx(2.0)

    def test_p95_of_ten_samples(self):
        tracker = LatencyTracker()
        for i in range(1, 11):
            tracker.record(float(i))
        # ceil(0.95 * 10) - 1 = 9 → sorted[9] = 10.0
        assert tracker.p95() == pytest.approx(10.0)

    def test_max_samples_rolling_window(self):
        tracker = LatencyTracker(max_samples=3)
        for v in [1.0, 2.0, 3.0, 4.0]:
            tracker.record(v)
        # Oldest (1.0) dropped; average of [2, 3, 4] = 3.0
        assert tracker.average() == pytest.approx(3.0)

    def test_single_sample_p95_equals_sample(self):
        tracker = LatencyTracker()
        tracker.record(7.7)
        assert tracker.p95() == pytest.approx(7.7)


class TestHelpers:
    def test_now_ts_returns_float(self):
        ts = now_ts()
        assert isinstance(ts, float)
        assert ts > 0

    def test_utc_now_returns_float(self):
        ts = utc_now()
        assert isinstance(ts, float)
        assert ts > 0

    def test_now_ts_is_monotonic(self):
        t1 = now_ts()
        t2 = now_ts()
        assert t2 >= t1

    def test_utc_now_is_wall_clock(self):
        import time as _time

        before = _time.time()
        ts = utc_now()
        after = _time.time()
        assert before <= ts <= after

    def test_clamp_within_range(self):
        assert clamp(5.0, 0.0, 10.0) == pytest.approx(5.0)

    def test_clamp_below_lo(self):
        assert clamp(-1.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_clamp_above_hi(self):
        assert clamp(15.0, 0.0, 10.0) == pytest.approx(10.0)
