from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class ArmHistoryPoint:
    week: int
    spend: float
    sales: float
    roas: float


def plot_roas_curves(history: dict[str, list[ArmHistoryPoint]], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for arm_id, points in history.items():
        xs = [p.week for p in points]
        ys = [p.roas for p in points]
        plt.plot(xs, ys, marker="o", label=arm_id)

    plt.title("ROAS by Arm Over Time")
    plt.xlabel("Week")
    plt.ylabel("ROAS")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    path = out / "roas_by_arm.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_spend_vs_sales(history: dict[str, list[ArmHistoryPoint]], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for arm_id, points in history.items():
        xs = [p.spend for p in points]
        ys = [p.sales for p in points]
        plt.scatter(xs, ys, label=arm_id)

    plt.title("Spend vs Sales by Arm")
    plt.xlabel("Spend")
    plt.ylabel("Sales")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    path = out / "spend_vs_sales.png"
    plt.savefig(path)
    plt.close()
    return path

