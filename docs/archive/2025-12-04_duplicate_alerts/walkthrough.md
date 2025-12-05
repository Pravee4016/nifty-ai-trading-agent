# Walkthrough: Alert Limits & Duplicate Detection Enhancement

## Changes Made

### 1. Increased Alert Limits
**File:** [config/settings.py](file:///Users/praveent/nifty-ai-trading-agent/config/settings.py#L208-L211)

| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| MAX_ALERTS_PER_DAY | 10 | **30** | +200% |
| MAX_ALERTS_PER_TYPE | 4 | **10** | +150% |
| MAX_ALERTS_PER_INSTRUMENT | 6 | **15** | +150% |

This provides much more flexibility while still maintaining risk control.

### 2. Enhanced Duplicate Detection
**File:** [main.py](file:///Users/praveent/nifty-ai-trading-agent/main.py#L540-L635)

#### Improvements:
1. **Cooldown Extended**: 30 minutes → **120 minutes (2 hours)**
   - Prevents same signal from repeating for 2 full hours
   
2. **Memory Retention**: 1 hour → **6 hours (full session)**
   - Remembers all alerts for entire trading day
   
3. **Price Tolerance Widened**: 0.05% → **0.07%**
   - Catches near-miss duplicates better (e.g., 26000 vs 26018)

### 3. Updated Documentation
**File:** [README.md](file:///Users/praveent/nifty-ai-trading-agent/README.md)

Updated Risk Management section and Configuration section to reflect new values.

---

## Impact

**Before:**
- Restrictive 10 alerts/day could miss valid setups on volatile days
- Duplicates could repeat after 30 minutes if price hovered at a level

**After:**
- 30 alerts/day allows capturing more opportunities
- Stronger duplicate suppression (2hr cooldown + 6hr memory)
- Better balance between opportunity and noise reduction

---

## Deployment

✅ Cloud Run Job updated successfully

The changes are now live and will take effect in the next trading session.
