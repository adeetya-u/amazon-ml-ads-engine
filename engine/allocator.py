from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AllocationConstraints:
    min_daily_spend_per_arm: float = 5.0
    max_weekly_fraction_per_arm: float = 0.40
    days_per_week: int = 7

    @property
    def min_weekly_spend_per_arm(self) -> float:
        return self.min_daily_spend_per_arm * self.days_per_week


def allocate_weekly_budget(
    *,
    weekly_budget: float,
    arm_ids: list[str],
    q_values: dict[str, float],
    constraints: AllocationConstraints,
) -> dict[str, float]:
    # This allocator is intentionally simple and deterministic
    # 1 apply a minimum floor so every active arm keeps collecting signal
    # 2 allocate remaining budget proportionally to bandit Q values
    # 3 enforce a hard cap per arm so a single winner does not swallow the week
    if weekly_budget < 0:
        raise ValueError("weekly_budget must be non-negative.")
    if not arm_ids:
        return {}

    min_floor = constraints.min_weekly_spend_per_arm
    cap = constraints.max_weekly_fraction_per_arm * weekly_budget if weekly_budget > 0 else 0.0

    # Start with floor allocations.
    alloc = {a: 0.0 for a in arm_ids}
    remaining = weekly_budget
    if min_floor > 0 and weekly_budget > 0:
        floor_total = min_floor * len(arm_ids)
        if floor_total >= weekly_budget:
            per_arm = weekly_budget / len(arm_ids)
            return {a: per_arm for a in arm_ids}
        for a in arm_ids:
            alloc[a] = min_floor
        remaining -= floor_total

    # Create non-negative weights derived from q-values.
    raw = {a: float(q_values.get(a, 0.0)) for a in arm_ids}
    min_raw = min(raw.values()) if raw else 0.0
    weights = {a: raw[a] - min_raw for a in arm_ids}  # >= 0
    if all(w == 0.0 for w in weights.values()):
        # All equal, or all missing, fall back to uniform weighting
        weights = {a: 1.0 for a in arm_ids}

    # Cap-aware proportional allocation (water-filling).
    active = set(arm_ids)
    while remaining > 1e-9 and active:
        wsum = sum(weights[a] for a in active)
        if wsum <= 0:
            break

        capped_any = False
        snapshot_remaining = remaining
        for a in list(active):
            share = snapshot_remaining * (weights[a] / wsum) if wsum > 0 else 0.0
            proposed = alloc[a] + share

            if cap > 0 and proposed > cap + 1e-12:
                delta = max(0.0, cap - alloc[a])
                alloc[a] += delta
                remaining -= delta
                active.remove(a)
                capped_any = True

        if capped_any:
            continue

        # No caps hit; safe to allocate remaining proportionally and finish.
        for a in list(active):
            alloc[a] += remaining * (weights[a] / wsum) if wsum > 0 else 0.0
        remaining = 0.0

    # If we still have tiny leftover (numerical), distribute without breaking caps.
    if remaining > 1e-6 and arm_ids:
        for a in arm_ids:
            if remaining <= 0:
                break
            room = cap - alloc[a] if cap > 0 else remaining
            if room <= 0:
                continue
            delta = min(room, remaining)
            alloc[a] += delta
            remaining -= delta

    return alloc

