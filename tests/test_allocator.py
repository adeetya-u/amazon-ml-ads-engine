from __future__ import annotations

from engine.allocator import AllocationConstraints, allocate_weekly_budget


def test_allocator_sums_to_budget() -> None:
    c = AllocationConstraints(min_daily_spend_per_arm=0.0, max_weekly_fraction_per_arm=1.0)
    alloc = allocate_weekly_budget(
        weekly_budget=100.0,
        arm_ids=["a", "b", "c"],
        q_values={"a": 1.0, "b": 2.0, "c": 3.0},
        constraints=c,
    )
    assert abs(sum(alloc.values()) - 100.0) < 1e-6


def test_allocator_floor_when_budget_too_small() -> None:
    c = AllocationConstraints(min_daily_spend_per_arm=5.0, max_weekly_fraction_per_arm=1.0)
    # min_weekly is 35 per arm; with 2 arms that's 70 > 60 so it should split evenly.
    alloc = allocate_weekly_budget(
        weekly_budget=60.0,
        arm_ids=["a", "b"],
        q_values={"a": 1.0, "b": 2.0},
        constraints=c,
    )
    assert abs(alloc["a"] - 30.0) < 1e-6
    assert abs(alloc["b"] - 30.0) < 1e-6


def test_allocator_respects_cap() -> None:
    c = AllocationConstraints(min_daily_spend_per_arm=0.0, max_weekly_fraction_per_arm=0.4)
    alloc = allocate_weekly_budget(
        weekly_budget=100.0,
        arm_ids=["a", "b", "c"],
        q_values={"a": 100.0, "b": 1.0, "c": 1.0},
        constraints=c,
    )
    assert alloc["a"] <= 40.0 + 1e-6
    assert abs(sum(alloc.values()) - 100.0) < 1e-6

