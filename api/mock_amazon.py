from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from engine.arms import Arm


@dataclass(frozen=True, slots=True)
class WeekPerformance:
    week: int
    arms: list[Arm]


def load_mock_campaign(path: str | Path) -> list[WeekPerformance]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))

    weeks: list[WeekPerformance] = []
    for week_obj in payload:
        week_num = int(week_obj["week"])
        arms = [
            Arm(
                id=a["id"],
                arm_type=a["arm_type"],
                spend=float(a["spend"]),
                sales=float(a["sales"]),
                impressions=int(a["impressions"]),
                clicks=int(a["clicks"]),
                orders=int(a["orders"]),
            )
            for a in week_obj["arms"]
        ]
        weeks.append(WeekPerformance(week=week_num, arms=arms))

    weeks.sort(key=lambda w: w.week)
    return weeks


def iter_weeks(path: str | Path, max_weeks: int | None = None) -> Iterable[WeekPerformance]:
    weeks = load_mock_campaign(path)
    if max_weeks is None:
        yield from weeks
        return
    yield from weeks[: max(0, int(max_weeks))]

