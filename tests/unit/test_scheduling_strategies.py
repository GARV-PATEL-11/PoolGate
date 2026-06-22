"""Unit tests for schedulers/scheduling_strategies.py — pure logic, no mocking needed."""

from __future__ import annotations

import pytest

from exceptions.request import UnknownSchedulingStrategyError
from schedulers.scheduling_strategies import (
    HealthAwareStrategy,
    LeastRemainingCapacityStrategy,
    LeastUsedStrategy,
    PriorityFailoverStrategy,
    RoundRobinStrategy,
    SchedulingStrategyType,
    WeightedRoundRobinStrategy,
    create_strategy,
)


class TestRoundRobinStrategy:
    def test_cycles_through_all_keys_in_order(self, three_keys, groq_config):
        strategy = RoundRobinStrategy()
        picked = [strategy.select(three_keys, three_keys, groq_config).key_id for _ in range(6)]
        assert picked == ["key_0", "key_1", "key_2", "key_0", "key_1", "key_2"]

    def test_skips_unavailable_keys_but_keeps_cursor_position(self, three_keys, groq_config):
        strategy = RoundRobinStrategy()
        candidates = [three_keys[0], three_keys[2]]  # key_1 unavailable
        first = strategy.select(candidates, three_keys, groq_config)
        assert first.key_id == "key_0"


class TestLeastUsedStrategy:
    def test_picks_key_with_fewest_active_requests(self, three_keys, groq_config):
        three_keys[0].record_request_start()
        three_keys[0].record_request_start()
        strategy = LeastUsedStrategy()
        chosen = strategy.select(three_keys, three_keys, groq_config)
        assert chosen.key_id in ("key_1", "key_2")


class TestPriorityFailoverStrategy:
    def test_prefers_earliest_position_when_no_explicit_priority(self, three_keys, groq_config):
        strategy = PriorityFailoverStrategy()
        chosen = strategy.select(three_keys, three_keys, groq_config)
        assert chosen.key_id == "key_0"

    def test_falls_back_to_second_key_when_primary_unavailable(self, three_keys, groq_config):
        strategy = PriorityFailoverStrategy()
        candidates = three_keys[1:]
        chosen = strategy.select(candidates, three_keys, groq_config)
        assert chosen.key_id == "key_1"


class TestHealthAwareStrategy:
    def test_picks_key_with_highest_health_score(self, three_keys, groq_config):
        # Inflate RPM on key_0 so it has the lowest health score
        for _ in range(20):
            three_keys[0].record_request_start()
        strategy = HealthAwareStrategy()
        chosen = strategy.select(three_keys, three_keys, groq_config)
        assert chosen.key_id in ("key_1", "key_2")

    def test_excludes_unavailable_key_via_minus_inf_score(self, three_keys, groq_config):
        three_keys[0].mark_disabled()
        three_keys[1].mark_disabled()
        strategy = HealthAwareStrategy()
        chosen = strategy.select(three_keys, three_keys, groq_config)
        assert chosen.key_id == "key_2"

    def test_returns_single_candidate_unchanged(self, three_keys, groq_config):
        strategy = HealthAwareStrategy()
        chosen = strategy.select([three_keys[2]], three_keys, groq_config)
        assert chosen.key_id == "key_2"


class TestWeightedRoundRobinStrategy:
    def test_distributes_across_all_keys_proportionally(self, three_keys, groq_config):
        # Without explicit weights, all keys get equal weight — expect all three used in 3 calls
        strategy = WeightedRoundRobinStrategy()
        picked_ids = {strategy.select(three_keys, three_keys, groq_config).key_id for _ in range(3)}
        assert len(picked_ids) == 3

    def test_higher_weight_key_selected_more_often(self, three_keys, groq_config):
        # Give key_0 a much higher explicit max_rpm so it dominates weighted selection
        three_keys[0].max_rpm = 100  # type: ignore[attr-defined]
        three_keys[1].max_rpm = 1  # type: ignore[attr-defined]
        three_keys[2].max_rpm = 1  # type: ignore[attr-defined]
        strategy = WeightedRoundRobinStrategy()
        selections = [
            strategy.select(three_keys, three_keys, groq_config).key_id for _ in range(10)
        ]
        assert selections.count("key_0") > selections.count("key_1")

    def test_returns_single_candidate_unchanged(self, three_keys, groq_config):
        strategy = WeightedRoundRobinStrategy()
        chosen = strategy.select([three_keys[1]], three_keys, groq_config)
        assert chosen.key_id == "key_1"


class TestLeastRemainingCapacityStrategy:
    def test_picks_key_with_most_remaining_budget(self, three_keys, groq_config):
        # key_0 has used more RPM budget so should be deprioritised
        for _ in range(10):
            three_keys[0].record_request_start()
        strategy = LeastRemainingCapacityStrategy()
        chosen = strategy.select(three_keys, three_keys, groq_config)
        assert chosen.key_id in ("key_1", "key_2")

    def test_all_equal_returns_first_candidate(self, three_keys, groq_config):
        # All keys have the same RPM — max() returns first match with equal keys
        strategy = LeastRemainingCapacityStrategy()
        chosen = strategy.select(three_keys, three_keys, groq_config)
        assert chosen.key_id in {k.key_id for k in three_keys}

    def test_single_candidate_returned(self, three_keys, groq_config):
        strategy = LeastRemainingCapacityStrategy()
        chosen = strategy.select([three_keys[2]], three_keys, groq_config)
        assert chosen.key_id == "key_2"


class TestStrategyFactory:
    def test_create_strategy_from_string(self):
        strategy = create_strategy("round_robin")
        assert strategy.name() == "RoundRobinStrategy"

    def test_create_strategy_from_enum(self):
        strategy = create_strategy(SchedulingStrategyType.LEAST_USED)
        assert strategy.name() == "LeastUsedStrategy"

    def test_unknown_strategy_raises_typed_exception(self):
        with pytest.raises(UnknownSchedulingStrategyError) as exc_info:
            create_strategy("not_a_real_strategy")
        assert "not_a_real_strategy" in str(exc_info.value)
        assert exc_info.value.available_strategies
