from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from analysis.visualizer import ArmHistoryPoint, plot_roas_curves, plot_spend_vs_sales
from api.mock_amazon import iter_weeks
from config import CampaignConfig, DEFAULT_CONFIG
from engine.allocator import AllocationConstraints, allocate_weekly_budget
from engine.bandit import EpsilonGreedyBandit
from engine.pruner import prune_arms
from engine.reward import normalize_roas


@dataclass
class EngineState:
    bandit: EpsilonGreedyBandit
    bad_weeks: dict[str, int]

    def to_dict(self) -> dict:
        return {"bandit": self.bandit.to_dict(), "bad_weeks": self.bad_weeks}

    @classmethod
    def from_dict(cls, payload: dict, *, seed: int | None = None) -> "EngineState":
        bandit = EpsilonGreedyBandit.from_dict(payload.get("bandit", {}), seed=seed)
        bad_weeks = {k: int(v) for k, v in (payload.get("bad_weeks") or {}).items()}
        return cls(bandit=bandit, bad_weeks=bad_weeks)


def load_state(path: Path, *, epsilon: float, seed: int | None) -> EngineState:
    if path.exists():
        # We always accept current epsilon from CLI so you can tune exploration without deleting state
        payload = json.loads(path.read_text(encoding="utf-8"))
        st = EngineState.from_dict(payload, seed=seed)
        st.bandit.epsilon = float(epsilon)
        return st
    return EngineState(bandit=EpsilonGreedyBandit(arm_ids=[], epsilon=epsilon, seed=seed), bad_weeks={})


def save_state(path: Path, state: EngineState) -> None:
    path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _debug_print(debug: bool, msg: str) -> None:
    if debug:
        print(msg)


def _record_history(history: dict[str, list[ArmHistoryPoint]], *, week: int, arms) -> None:
    for a in arms:
        history.setdefault(a.id, []).append(ArmHistoryPoint(week=week, spend=a.spend, sales=a.sales, roas=a.roas))


def _print_week_summary(*, week: int, active_arm_ids: list[str], pruned_arm_ids: list[str], qvals: dict[str, float], alloc: dict[str, float]) -> None:
    print(f"\nWeek {week}")
    print(f"Active arms: {len(active_arm_ids)} | Pruned: {pruned_arm_ids or 'none'}")
    for arm_id in active_arm_ids:
        q = qvals.get(arm_id, 0.0)
        a = alloc.get(arm_id, 0.0)
        print(f"  - {arm_id:20s} Q={q:.3f} alloc=${a:.2f}")


def run(
    config: CampaignConfig,
    *,
    weeks: int,
    plot: bool,
    state_path: Path,
    seed: int | None,
    debug: bool,
) -> None:
    """
    Run a week by week simulation using mock Amazon Ads performance data
    Bandit and pruning state is persisted to a JSON file so repeated runs can continue learning
    """
    state = load_state(state_path, epsilon=config.epsilon, seed=seed)

    history: dict[str, list[ArmHistoryPoint]] = {}

    for wk in iter_weeks("data/mock_campaign.json", max_weeks=weeks):
        arms = wk.arms
        active_arm_ids = [a.id for a in arms]
        state.bandit.ensure_arms(active_arm_ids)

        _debug_print(debug, f"DEBUG week={wk.week} active_arms={active_arm_ids}")

        # Reward is normalized ROAS for the week so arms are comparable on a stable 0 to 1 scale
        rewards = normalize_roas(arms)
        for arm_id, r in rewards.items():
            state.bandit.update(arm_id, r)

        # Pruning happens after the update so an arm still gets credit for the current week
        decision = prune_arms(
            week=wk.week,
            arms=arms,
            prune_threshold=config.prune_threshold,
            bad_weeks=state.bad_weeks,
            log_path="pruned_log.json",
        )
        state.bad_weeks = decision.updated_bad_weeks
        if decision.pruned_arm_ids:
            state.bandit.remove_arms(decision.pruned_arm_ids)

        pruned_set = set(decision.pruned_arm_ids)
        active_arm_ids = [a.id for a in arms if a.id not in pruned_set]

        qvals = state.bandit.q_values(active_arm_ids)
        if debug:
            rounded_q = {k: round(v, 4) for k, v in qvals.items()}
            _debug_print(debug, f"DEBUG week={wk.week} q_values={rounded_q}")

        # Allocate the weekly budget using floors and caps to keep exploration healthy
        alloc = allocate_weekly_budget(
            weekly_budget=config.weekly_budget,
            arm_ids=active_arm_ids,
            q_values=qvals,
            constraints=AllocationConstraints(
                min_daily_spend_per_arm=config.min_daily_spend_per_arm,
                max_weekly_fraction_per_arm=config.max_weekly_fraction_per_arm,
            ),
        )
        if debug:
            rounded_alloc = {k: round(v, 2) for k, v in alloc.items()}
            _debug_print(debug, f"DEBUG week={wk.week} allocation={rounded_alloc}")

        _record_history(history, week=wk.week, arms=arms)
        _print_week_summary(
            week=wk.week,
            active_arm_ids=active_arm_ids,
            pruned_arm_ids=decision.pruned_arm_ids,
            qvals=qvals,
            alloc=alloc,
        )

        save_state(state_path, state)

    if plot:
        out_dir = Path("output")
        roas_path = plot_roas_curves(history, out_dir)
        spend_sales_path = plot_spend_vs_sales(history, out_dir)
        print(f"\nSaved plots: {roas_path} , {spend_sales_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Epsilon-greedy MAB simulation for Amazon Ads optimization.")
    p.add_argument("--weeks", type=int, default=4)
    p.add_argument("--budget", type=float, default=DEFAULT_CONFIG.weekly_budget)
    p.add_argument("--epsilon", type=float, default=DEFAULT_CONFIG.epsilon)
    p.add_argument("--prune-threshold", type=float, default=DEFAULT_CONFIG.prune_threshold)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--state-path", type=str, default="state.json")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CampaignConfig(
        weekly_budget=float(args.budget),
        epsilon=float(args.epsilon),
        prune_threshold=float(args.prune_threshold),
        min_daily_spend_per_arm=DEFAULT_CONFIG.min_daily_spend_per_arm,
        max_weekly_fraction_per_arm=DEFAULT_CONFIG.max_weekly_fraction_per_arm,
    )
    # keep debug as a CLI flag so you can flip it on without editing code
    # using print here is intentional for local scripts and quick iteration
    run(
        cfg,
        weeks=int(args.weeks),
        plot=bool(args.plot),
        state_path=Path(args.state_path),
        seed=int(args.seed),
        debug=bool(args.debug),
    )


if __name__ == "__main__":
    main()

