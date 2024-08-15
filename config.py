from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CampaignConfig:
    weekly_budget: float = 50.0
    epsilon: float = 0.1
    prune_threshold: float = 0.85
    min_daily_spend_per_arm: float = 5.0
    max_weekly_fraction_per_arm: float = 0.40


DEFAULT_CONFIG = CampaignConfig()

