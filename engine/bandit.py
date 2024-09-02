from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BanditArmState:
    q_value: float = 0.0
    n_pulls: int = 0

    def update(self, reward: float) -> None:
        # Incremental mean update so we do not store reward history
        self.n_pulls += 1
        self.q_value = self.q_value + (1.0 / self.n_pulls) * (reward - self.q_value)


class EpsilonGreedyBandit:
    def __init__(self, arm_ids: list[str], epsilon: float = 0.1, seed: int | None = None):
        # epsilon is exploration rate
        # seed is here purely for deterministic tests and repeatable sims
        self.epsilon = float(epsilon)
        self._rng = np.random.default_rng(seed)
        self.state: dict[str, BanditArmState] = {arm_id: BanditArmState() for arm_id in arm_ids}

    def ensure_arms(self, arm_ids: list[str]) -> None:
        # Add unseen arms with zero priors so the bandit can warm up naturally
        for arm_id in arm_ids:
            if arm_id not in self.state:
                self.state[arm_id] = BanditArmState()

    def remove_arms(self, arm_ids: list[str]) -> None:
        for arm_id in arm_ids:
            self.state.pop(arm_id, None)

    def select(self, active_arm_ids: list[str]) -> str:
        if not active_arm_ids:
            raise ValueError("No active arms to select from.")

        self.ensure_arms(active_arm_ids)
        # Explore with probability epsilon, otherwise exploit current best estimate
        explore = self._rng.random() < self.epsilon
        if explore:
            return str(self._rng.choice(active_arm_ids))

        # Break ties by first max, deterministic given stable list order
        best_id = max(active_arm_ids, key=lambda a: self.state[a].q_value)
        return best_id

    def update(self, arm_id: str, reward: float) -> None:
        if arm_id not in self.state:
            self.state[arm_id] = BanditArmState()
        self.state[arm_id].update(float(reward))

    def q_values(self, active_arm_ids: list[str]) -> dict[str, float]:
        self.ensure_arms(active_arm_ids)
        return {a: self.state[a].q_value for a in active_arm_ids}

    def to_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "arms": {arm_id: {"q_value": s.q_value, "n_pulls": s.n_pulls} for arm_id, s in self.state.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict, *, seed: int | None = None) -> "EpsilonGreedyBandit":
        arm_ids = list((payload.get("arms") or {}).keys())
        bandit = cls(arm_ids=arm_ids, epsilon=float(payload.get("epsilon", 0.1)), seed=seed)
        bandit.state = {
            arm_id: BanditArmState(
                q_value=float(s.get("q_value", 0.0)),
                n_pulls=int(s.get("n_pulls", 0)),
            )
            for arm_id, s in (payload.get("arms") or {}).items()
        }
        return bandit

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, seed: int | None = None) -> "EpsilonGreedyBandit":
        p = Path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(payload, seed=seed)

