from __future__ import annotations

from collections.abc import Iterable

from engine.arms import Arm


def normalize_roas(arms: Iterable[Arm]) -> dict[str, float]:
    arms_list = list(arms)
    roas_values = [a.roas for a in arms_list]
    if not roas_values:
        return {}

    r_min = min(roas_values)
    r_max = max(roas_values)
    if r_max == r_min:
        return {a.id: 0.5 for a in arms_list}

    denom = r_max - r_min
    return {a.id: (a.roas - r_min) / denom for a in arms_list}

