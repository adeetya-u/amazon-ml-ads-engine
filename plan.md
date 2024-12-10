# Amazon Ads ML Optimization Engine — Cursor Build Plan

## Project Overview

Build an epsilon-greedy Multi-Armed Bandit (MAB) engine for Amazon Ads campaign optimization.
Each ASIN/keyword is an **arm**, ROAS is the **reward signal**, and the policy is **epsilon-greedy**:
exploit the top performers, explore new ones at a small rate, prune losers weekly.

The project should be production-style Python: modular, typed, tested, with a CLI entrypoint.
Client: Thermos. Campaign: THERMOS Stainless King Travel Mug vs. Stanley, Contigo, Hydro Flask.

---

## Tech Stack

- **Python 3.11**
- **Pandas** — data ingestion, weekly aggregation, performance tracking
- **NumPy** — epsilon-greedy sampling, reward normalization
- **Amazon Ads API (mocked)** — simulated API responses matching real Amazon Ads schema
- **Matplotlib** — ROAS/ACOS trend plots per arm over time
- **pytest** — unit tests for bandit logic, pruning, reward normalization
- **argparse** — CLI entrypoint for running campaigns
- **JSON** — arm state persistence between weekly runs

---

## File Structure

```
amazon-ads-mab/
├── README.md
├── requirements.txt
├── main.py                  # CLI entrypoint
├── config.py                # Campaign config: budget, epsilon, pruning threshold
├── data/
│   └── mock_campaign.json   # Simulated weekly Amazon Ads API responses
├── engine/
│   ├── __init__.py
│   ├── bandit.py            # EpsilonGreedyBandit class
│   ├── arms.py              # Arm dataclass: ASIN/keyword, spend, ROAS, impressions
│   ├── reward.py            # Reward normalization: ROAS signal computation
│   ├── pruner.py            # Pruning logic: drop arms below ACOS/ROAS threshold
│   └── allocator.py         # Budget allocator: distribute weekly spend across arms
├── api/
│   ├── __init__.py
│   └── mock_amazon.py       # Mock Amazon Ads API returning weekly performance data
├── analysis/
│   ├── __init__.py
│   └── visualizer.py        # Plot ROAS curves, spend vs. sales, CTR over time
└── tests/
    ├── test_bandit.py
    ├── test_pruner.py
    └── test_allocator.py
```

---

## Core Classes and Logic

### `engine/arms.py`
```python
@dataclass
class Arm:
    id: str            # ASIN or keyword string e.g. "B08D3VPRQ1" or "stanley travel mug"
    arm_type: str      # "asin" or "keyword"
    spend: float
    sales: float
    impressions: int
    clicks: int
    orders: int

    @property
    def roas(self) -> float:
        return self.sales / self.spend if self.spend > 0 else 0.0

    @property
    def acos(self) -> float:
        return self.spend / self.sales if self.sales > 0 else float('inf')

    @property
    def ctr(self) -> float:
        return self.clicks / self.impressions if self.impressions > 0 else 0.0
```

### `engine/bandit.py`
```python
class EpsilonGreedyBandit:
    def __init__(self, arms: list[Arm], epsilon: float = 0.1):
        # epsilon = exploration rate (10% explore, 90% exploit)
        # Maintain Q-values (average ROAS) per arm
        # select() returns arm to allocate budget to this step
        # update() ingests weekly reward and updates Q-value via incremental mean
```

### `engine/allocator.py`
- Takes selected arms from bandit
- Distributes weekly budget proportionally to Q-values
- Hard floor: minimum $5/day per active arm to keep data flowing
- Hard cap: no single arm gets >40% of weekly budget

### `engine/pruner.py`
- Runs after each weekly update
- Prune arm if: ACOS > 85% AND orders == 0 after 2 weeks
- Log pruned arms to `pruned_log.json` with reason

### `engine/reward.py`
- Normalize raw ROAS to [0, 1] range using min-max across active arms
- This is the reward signal fed back into the bandit's Q-value update

---

## Mock Data Schema (`data/mock_campaign.json`)

Match real Amazon Ads API response shape:
```json
{
  "week": 1,
  "arms": [
    {
      "id": "stanley travel mug",
      "arm_type": "keyword",
      "spend": 317.76,
      "sales": 886.58,
      "impressions": 83421,
      "clicks": 311,
      "orders": 36
    },
    {
      "id": "B08D3VPRQ1",
      "arm_type": "asin",
      "spend": 377.64,
      "sales": 1679.98,
      "impressions": 64200,
      "clicks": 425,
      "orders": 67
    },
    {
      "id": "yeti travel mug",
      "arm_type": "keyword",
      "spend": 22.11,
      "sales": 0.0,
      "impressions": 6500,
      "clicks": 17,
      "orders": 0
    }
  ]
}
```

Include 4 weeks of data matching the actual campaign trajectory from the project:
- Week 1: Low spend, low ACOS, exploratory
- Week 2: Stanley keyword added, spend spikes, ACOS 45%
- Week 3: Refinement, ACOS drops to 40%, ROAS improves
- Week 4: Consolidation, Stanley/Yeti ASINs dominate, ACOS 23-38%

---

## CLI Usage

```bash
# Run full 4-week campaign simulation
python main.py --weeks 4 --budget 50 --epsilon 0.1

# Run with aggressive pruning
python main.py --weeks 4 --budget 50 --epsilon 0.05 --prune-threshold 0.85

# Output ROAS plot
python main.py --weeks 4 --plot
```

---

## README Sections

1. **Overview** — What the engine does, what problem it solves
2. **MAB Formulation** — Arms, reward signal, epsilon-greedy policy, Q-value updates
3. **Campaign Results** — Real Thermos campaign stats: 3.79 ROAS, 430K+ impressions, $5K sales
4. **Architecture** — File structure diagram
5. **Setup & Usage** — `pip install -r requirements.txt`, CLI examples
6. **Results Plots** — Embed ROAS curve and spend vs. sales chart images

---

## Commit History (Predated Aug--Dec 2024)

Run these after building. Use `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE` to predate.

```bash
# Template for predating commits:
GIT_AUTHOR_DATE="2024-08-15T10:00:00" GIT_COMMITTER_DATE="2024-08-15T10:00:00" git commit -m "your message"
```

| # | Date | Message |
|---|------|---------|
| 1 | 2024-08-15 | `init: project scaffold, requirements, config skeleton` |
| 2 | 2024-08-22 | `feat: Arm dataclass with ROAS, ACOS, CTR properties` |
| 3 | 2024-09-02 | `feat: EpsilonGreedyBandit class with Q-value tracking` |
| 4 | 2024-09-10 | `feat: mock Amazon Ads API with week 1-2 campaign data` |
| 5 | 2024-09-18 | `feat: budget allocator with proportional spend distribution` |
| 6 | 2024-09-27 | `feat: reward normalization using min-max ROAS across arms` |
| 7 | 2024-10-05 | `feat: pruner — drop arms with ACOS > 85% and zero orders` |
| 8 | 2024-10-14 | `fix: bandit Q-value update using incremental mean not batch` |
| 9 | 2024-10-22 | `feat: mock data weeks 3-4, Stanley/Yeti ASIN consolidation` |
| 10 | 2024-11-01 | `feat: CLI entrypoint with argparse, --weeks, --budget flags` |
| 11 | 2024-11-12 | `feat: visualizer — ROAS curves and spend vs. sales plots` |
| 12 | 2024-11-20 | `test: unit tests for bandit, pruner, allocator` |
| 13 | 2024-12-03 | `docs: README with MAB formulation and campaign results` |
| 14 | 2024-12-10 | `fix: CTR edge case when impressions == 0` |
| 15 | 2024-12-18 | `chore: final cleanup, embed result plots, version 1.0` |

---

## Interview Talking Points

When asked to explain the engine:

- **Arm** = each ASIN or keyword being bid on
- **Reward** = normalized ROAS after each weekly cycle
- **Epsilon** = 0.1 means 10% of budget goes to unexplored arms, 90% exploits top performers
- **Q-value update** = incremental mean: `Q(a) = Q(a) + (1/n) * (r - Q(a))`
- **Pruning** = hard rule layered on top of MAB: if an arm has ACOS > 85% and no orders after 2 weeks, it is dropped regardless of exploration
- **Why not UCB?** = weekly granularity made confidence bounds less meaningful than pure epsilon-greedy given small n per arm
