from __future__ import annotations

from engine.arms import Arm
from engine.pruner import prune_arms


def test_pruner_requires_two_bad_weeks(tmp_path) -> None:
    log_path = tmp_path / "pruned_log.json"
    bad_weeks: dict[str, int] = {}

    arms_week1 = [
        Arm(
            id="bad",
            arm_type="keyword",
            spend=100.0,
            sales=0.0,
            impressions=1000,
            clicks=10,
            orders=0,
        )
    ]
    d1 = prune_arms(
        week=1,
        arms=arms_week1,
        prune_threshold=0.85,
        bad_weeks=bad_weeks,
        log_path=log_path,
    )
    assert d1.pruned_arm_ids == []
    assert d1.updated_bad_weeks["bad"] == 1
    assert not log_path.exists()

    arms_week2 = arms_week1
    d2 = prune_arms(
        week=2,
        arms=arms_week2,
        prune_threshold=0.85,
        bad_weeks=d1.updated_bad_weeks,
        log_path=log_path,
    )
    assert d2.pruned_arm_ids == ["bad"]
    assert log_path.exists()


def test_pruner_resets_on_good_week(tmp_path) -> None:
    log_path = tmp_path / "pruned_log.json"
    bad_weeks = {"a": 1}
    arms = [
        Arm(
            id="a",
            arm_type="asin",
            spend=10.0,
            sales=100.0,
            impressions=100,
            clicks=5,
            orders=1,
        )
    ]
    d = prune_arms(week=2, arms=arms, prune_threshold=0.85, bad_weeks=bad_weeks, log_path=log_path)
    assert d.pruned_arm_ids == []
    assert d.updated_bad_weeks["a"] == 0

