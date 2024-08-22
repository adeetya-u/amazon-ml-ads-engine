from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Arm:
    id: str  # ASIN or keyword string
    arm_type: str  # "asin" or "keyword"
    spend: float
    sales: float
    impressions: int
    clicks: int
    orders: int

    @property
    def roas(self) -> float:
        return (self.sales / self.spend) if self.spend > 0 else 0.0

    @property
    def acos(self) -> float:
        return (self.spend / self.sales) if self.sales > 0 else float("inf")

    @property
    def ctr(self) -> float:
        return (self.clicks / self.impressions) if self.impressions > 0 else 0.0

