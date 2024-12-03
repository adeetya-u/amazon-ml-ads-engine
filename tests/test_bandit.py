from __future__ import annotations

from engine.bandit import EpsilonGreedyBandit


def test_incremental_mean_update() -> None:
    b = EpsilonGreedyBandit(arm_ids=["a"], epsilon=0.0, seed=1)
    b.update("a", 1.0)
    assert b.state["a"].q_value == 1.0
    b.update("a", 0.0)
    assert b.state["a"].n_pulls == 2
    assert b.state["a"].q_value == 0.5


def test_select_exploit_picks_best_arm() -> None:
    b = EpsilonGreedyBandit(arm_ids=["a", "b"], epsilon=0.0, seed=2)
    b.state["a"].q_value = 0.1
    b.state["b"].q_value = 0.9
    assert b.select(["a", "b"]) == "b"


def test_select_explore_uniformish() -> None:
    b = EpsilonGreedyBandit(arm_ids=["a", "b", "c"], epsilon=1.0, seed=3)
    picks = [b.select(["a", "b", "c"]) for _ in range(30)]
    # With seeded RNG and 30 draws, we should see >1 unique arm.
    assert len(set(picks)) > 1

