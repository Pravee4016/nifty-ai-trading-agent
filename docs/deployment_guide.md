# Deployment & Monitoring Guide

## Quick Deploy

```bash
# 1. Set project
export GOOGLE_CLOUD_PROJECT=your-project-id

# 2. Deploy
bash deploy.sh
```

## Monitoring Commands

### Check Market State in Logs
```bash
# Real-time monitoring
gcloud functions logs read nifty-trading-agent --region=asia-south1 --limit=50 | grep "Market State"

# Expected output:
# 📊 Market State: CHOPPY | Confidence: 75% | Reasons: Range compressed...
# 📊 Market State: EXPANSIVE | Confidence: 100% | Reasons: Large bodies...
```

### Check Blocked Signals
```bash
gcloud functions logs read nifty-trading-agent --region=asia-south1 --limit=50 | grep "Blocked"

# Expected:
# 🛑 CHOPPY State | Blocking all 6 signals
# ⏭️ TRANSITION State | Blocked 2 signals (state-gated)
```

### Check ML Decisions
```bash
gcloud functions logs read nifty-trading-agent --region=asia-south1 --limit=50 | grep "ML"

# Expected:
# ✅ ML Accepted | Prob: 72% (EXPANSIVE) | BULLISH_BREAKOUT
# 🛑 ML Rejected | Prob: 58% < 65% | EXPANSIVE | RETEST
```

## State Distribution Tracking

Create this script to analyze state distribution:

```python
# scripts/analyze_states.py
import re
from collections import Counter

# Get logs
logs = """paste logs here"""

states = re.findall(r'Market State: (\w+)', logs)
counter = Counter(states)

total = sum(counter.values())
print(f"State Distribution (n={total}):")
for state, count in counter.most_common():
    print(f"  {state}: {count} ({count/total:.0%})")
```

## Rollback if Needed

```bash
# Option 1: Disable Market State Engine (keep ML)
gcloud functions deploy nifty-trading-agent \
  --update-env-vars USE_MARKET_STATE_ENGINE=False

# Option 2: Full rollback to previous version  
gcloud functions deploy nifty-trading-agent \
  --set-env-vars USE_ML_FILTERING=False
```

## Enable ML Later

Once you have 100+ training samples:

```bash
# 1. Train model
python scripts/train_lgbm_model.py

# 2. Enable ML
gcloud functions deploy nifty-trading-agent \
  --update-env-vars USE_ML_FILTERING=True
```
