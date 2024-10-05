from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from engine.arms import Arm


@dataclass
class PruneDecision:
    pruned_arm_ids: list[str]
    updated_bad_weeks: dict[str, int]


def prune_arms(
    *,
    week: int,
    arms: list[Arm],
    prune_threshold: float,
    bad_weeks: dict[str, int],
    log_path: str | Path,
) -> PruneDecision:
    # We track consecutive bad weeks so pruning is not triggered by a single noisy week
    # bad means high ACOS and zero orders for the week
    updated_bad = dict(bad_weeks)
    pruned: list[str] = []

    for a in arms:
        is_bad = (a.acos > prune_threshold) and (a.orders == 0)
        if is_bad:
            updated_bad[a.id] = int(updated_bad.get(a.id, 0)) + 1
        else:
            updated_bad[a.id] = 0

        if updated_bad[a.id] >= 2:
            pruned.append(a.id)

    if pruned:
        _append_pruned_log(
            log_path=log_path,
            week=week,
            pruned_arm_ids=pruned,
            reason=f"ACOS>{prune_threshold} and orders==0 for 2 weeks",
        )

    # remove counters for arms not present this week (keep file clean)
    present = {a.id for a in arms}
    updated_bad = {arm_id: cnt for arm_id, cnt in updated_bad.items() if arm_id in present}

    return PruneDecision(pruned_arm_ids=pruned, updated_bad_weeks=updated_bad)


def _append_pruned_log(*, log_path: str | Path, week: int, pruned_arm_ids: list[str], reason: str) -> None:
    p = Path(log_path)
    if p.exists():
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            payload = []
    else:
        payload = []

    payload.append({"week": int(week), "pruned_arm_ids": list(pruned_arm_ids), "reason": str(reason)})
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

