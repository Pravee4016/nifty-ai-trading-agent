# NIFTY AI Trading Agent - Session Notes & Roadmap
**Date**: December 4, 2025  
**Status**: Production Ready with Enhancement Roadmap

---

## Quick Reference

- **Current Rating**: 8/10 for 5min Nifty intraday
- **Goal**: 9/10 - Consistent 20-30 point captures (1000-1500 Rs per option)
- **Next Session**: Dec 5, 2025 - Monitor and validate recent changes

---

## Recent Changes Deployed (Dec 4, 2025)

### ✅ Fixes
1. **Event Tracking** - All signal types (PIN_BAR, ENGULFING, INSIDE_BAR) now tracked for EOD summary
2. **Market Hours Spam** - Removed "Outside market hours" message spam after 3:30 PM
3. **Choppy Session Filter** - Fixed broken filter (added `get_historical_data` method to fetcher)
4. **datetime Import** - Fixed time-of-day filter failures in technical.py

### ✅ Enhancements
1. **Alert Limits** - Removed daily cap (was 10, now unlimited with other filters)
2. **Duplicate Detection** - Enhanced with:
   - 30min cooldown (was 20min)
   - 0.1% price tolerance (was 0.07%)
   - Level-based memory (blocks same S/R level all day)
3. **Backtest** - Now uses production logic for accurate testing

---

## Dec 5 Monitoring Checklist

### Expected Behavior
- ✅ 10-15 alerts during trading hours
- ✅ No duplicate/spam alerts
- ✅ EOD summary with non-zero event counts
- ✅ No messages after market close

### Warning Signs
- ❌ >30 alerts (per-type limits not working)
- ❌ Duplicate alerts (level memory failing)
- ❌ All zeros in EOD (event tracking broken)
- ❌ Spam messages (market hours check failing)

---

## Phase 1 Implementation Plan (Post-Monitoring)

**Goal**: 8/10 → 8.5/10  
**Target**: 5-8 quality signals per day  
**Estimated Time**: 2-3 hours

### Critical Changes

#### 1. Fix Legacy Helper Bug ❌
**File**: `analysis_module/technical.py`  
**Issue**: `analyze_instrument()` calls `detect_inside_bar(df)` without context parameters  
**Fix**: Update call or delete if unused

#### 2. Integrate Choppy Session Filter ⚠️
**File**: `analysis_module/technical.py`  
**Change**: Call `_is_choppy_session()` in `analyze_with_multi_tf()` BEFORE pattern detection
```python
is_choppy, reason = self._is_choppy_session(df)
if is_choppy:
    # Skip continuation patterns (breakout, inside bar)
    # OR reduce confidence by 20 points
```

#### 3. Add Opening Range Breakout (ORB) ⭐
**File**: `analysis_module/technical.py`  
**Add**: 
- `get_opening_range(df, duration_mins=15)` method
- Integrate ORB into `detect_breakout()`
- Include ORB levels in support/resistance
- Add to `get_higher_tf_context()`

**Why ORB?**: Most reliable Nifty intraday setup for 20-30 point moves

---

## Future Enhancements (Phase 2)

### For 20-30 Point Option Scalping

1. **Minimum Distance to Levels** - Avoid "into-the-wall" trades
   - Require 0.25-0.3% distance to opposite level (~65-80 points)

2. **Partial Profit / Trailing** - Lock in gains
   - Book 50% at +15-20 points
   - Trail remaining with SL to cost or VWAP/EMA

3. **Option Delta Awareness** - Bridge spot to premium
   - Approximate ATM delta ~0.45-0.55
   - Ensure spot TP translates to 1000-1500 Rs

4. **Backtest-Driven Refinement** - Optimize patterns
   - Track: time-of-day, distance to levels, ATR%, session regime
   - Find best 1-3 setups for Nifty

---

## System Architecture

### Signal Flow
```
Data Fetch (NSE + yfinance)
    ↓
Preprocessing (OHLCV + indicators)
    ↓
Higher TF Context (15m trend, PDH/PDL, VWAP, EMA)
    ↓
Pattern Detection (Breakout, Retest, Inside Bar, Pin Bar, Engulfing)
    ↓
Filters Applied:
  - Confidence gate (≥65%)
  - Choppy session detection
  - Correlation check (max 3 same direction in 15min)
  - Duplicate detection (30min + level memory)
  - Conflict filter (15min opposing signals)
  - Time-of-day filter
  - Per-type limit (10/day)
  - Per-instrument limit (15/day)
    ↓
Signal Generation
    ↓
Telegram Alert + Firestore Logging
```

### Key Filters (8 Active)
1. ✅ Per-Type Limit: Max 10 per signal type
2. ✅ Per-Instrument Limit: Max 15 per instrument
3. ✅ Duplicate Detection: 30min cooldown, 0.1% tolerance, level memory
4. ✅ Conflict Filter: 15min opposing signal block
5. ✅ Choppy Session: Blocks low volatility signals
6. ✅ Correlation Check: Max 3 same-direction in 15min
7. ✅ Time-of-Day: Avoids first 15min, lunch, last hour
8. ✅ Confidence Gate: Minimum 65%

---

## Expert Feedback Summary

**Overall Assessment**: 8/10

### Strengths
- Multi-filter context (Trend + PDH/PDL + VWAP + EMA + S/R + ATR + RSI + volume)
- Good R:R discipline (reject if <1.5, ATR-based targets)
- Clear price-action patterns with context
- Comprehensive noise filters

### Gaps
1. Legacy helper mismatch (will cause errors)
2. No explicit profit-booking/trailing
3. No opening range logic
4. Choppy filter not integrated into signals
5. No pattern-specific minimum distance to levels

### To Reach 9/10
- Position-sizing + option-greeks awareness
- Opening range + session regime handling
- Focus on best 1-3 opportunities per day

---

## Backtest Results (Nov 26-28, 2025)

### Raw Results
- **507 total signals** (169/day)
- Breakdown:
  - RESISTANCE_BOUNCE: 259 (51%)
  - SUPPORT_BOUNCE: 159 (31%)
  - Pin Bars: 54 (11%)
  - Engulfing: 35 (7%)

### Analysis
- **Market**: Range-bound consolidation
- **Issue**: Constant bounces between S/R → signal spam
- **Production**: Should reduce to 10-15/day with per-type/instrument limits

### Post-Phase 1 Expected
- **Fewer signals**: 169 → 5-8 per day
- **Better timing**: ORB + choppy filter
- **Higher quality**: Focus on best setups

---

## File Locations

### Project Files
- Main: `/Users/praveent/nifty-ai-trading-agent/main.py`
- Technical: `/Users/praveent/nifty-ai-trading-agent/analysis_module/technical.py`
- Fetcher: `/Users/praveent/nifty-ai-trading-agent/data_module/fetcher.py`
- Config: `/Users/praveent/nifty-ai-trading-agent/config/settings.py`

### Documentation
- This file: `/Users/praveent/nifty-ai-trading-agent/NOTES.md`
- Implementation Plan: `/Users/praveent/.gemini/antigravity/brain/.../implementation_plan.md`
- Roadmap: `/Users/praveent/.gemini/antigravity/brain/.../improvement_roadmap.md`
- Verification: `/Users/praveent/.gemini/antigravity/brain/.../production_verification_checklist.md`

---

## Deployment

### Current Setup
- **Platform**: Google Cloud Run Job
- **Trigger**: Cloud Scheduler (every 5 minutes during market hours)
- **Deploy Script**: `./deploy_job.sh`
- **Last Deployed**: Dec 4, 2025 ~21:00 IST

### Quick Deploy
```bash
cd /Users/praveent/nifty-ai-trading-agent
./deploy_job.sh
```

---

## Contact & Support

For implementation questions, refer to:
1. Implementation Plan artifact
2. Improvement Roadmap artifact  
3. This NOTES.md file

**Last Updated**: December 4, 2025, 21:49 IST
