# Amazon Ads ML Optimization Engine

An epsilon-greedy Multi-Armed Bandit (MAB) simulation for Amazon Ads campaign optimization.

## What it does
- **Arms**: each ASIN or keyword being bid on
- **Reward**: normalized ROAS (min-max across active arms per week)
- **Policy**: epsilon-greedy (explore at rate `epsilon`, otherwise exploit best estimated arm)
- **Pruning**: drop arms with **ACOS > threshold** and **0 orders** for **2 weeks**
- **Persistence**: bandit state + prune counters saved to JSON between runs

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# Run full 4-week campaign simulation
python main.py --weeks 4 --budget 50 --epsilon 0.1

# Run with aggressive pruning
python main.py --weeks 4 --budget 50 --epsilon 0.05 --prune-threshold 0.85

# Output plots
python main.py --weeks 4 --plot
```

## Testing

```bash
pytest
```

