# Production Verification Checklist - Dec 5, 2025

## Pre-Market (Before 9:15 AM)

- [ ] Cloud Run Job scheduler active
- [ ] Last deployment: Dec 4, 2025 ~21:00 IST ✅
- [ ] All changes deployed:
  - ✅ 30min duplicate cooldown
  - ✅ 0.1% price tolerance
  - ✅ Level-based memory
  - ✅ Choppy session filter
  - ✅ datetime import fix

---

## During Market Hours (9:15 AM - 3:30 PM)

### ✅ Expected Behavior

1. **Alerts**:
   - Should see **10-15 alerts** for the day (max)
   - No spam or repeated alerts at same levels
   - No "Outside market hours" messages during trading

2. **Alert Quality**:
   - Confidence ≥ 65%
   - Variety of signal types (not all RESISTANCE_BOUNCE)
   - Proper entry/SL/TP levels

3. **End-of-Day Summary** (3:25-3:30 PM):
   - **Today's Events** should show non-zero counts
   - Breakouts, retests, pin bars, engulfing all tracked
   - Should match actual alerts received

### ❌ Warning Signs

- **Too many alerts** (>30): Per-type/instrument limits not working
- **Duplicate alerts**: Level memory or cooldown failing
- **All zeros in EOD**: Event tracking broken again
- **Spam messages**: Market hours check failing

---

## Post-Market (After 3:30 PM)

- [ ] Check EOD summary received ~3:25-3:30 PM
- [ ] Verify event counts match received alerts
- [ ] **No "Outside market hours" spam** after 3:30 PM
- [ ] Check Cloud Run Job logs if needed

---

## What Should Happen Tomorrow

### Morning (9:15-12:00)
- First alerts should arrive within 15-30 minutes
- Should see mix of signal types if market is active
- Duplicate detection blocks repeats at same level

### Afternoon (12:00-3:30)
- Alert rate slows if per-type limits hit
- Level memory prevents same-level repeats
- Choppy filter blocks if market goes flat

### End of Day (3:25-3:30)
- EOD summary with actual event counts
- Performance tracking updated in Firestore
- **No spam messages after 3:30 PM**

---

## If Issues Found

### Too Many Alerts (>30)
1. Check which signal type is dominant
2. May need to tighten level rounding or cooldown
3. Check if per-type limits are being enforced

### Duplicate Alerts
1. Verify level memory is working (check logs)
2. May need tighter price tolerance (0.1% → 0.05%)
3. Check if instrument names match exactly

### No Alerts At All
1. Choppy filter may be too aggressive
2. Check market volatility (low volatility day?)
3. Review Cloud Run Job logs for errors

### EOD Still Shows Zeros
1. Check event tracking in `_track_daily_event`
2. Verify persistence.add_event is being called
3. Check Firestore for daily stats document

---

## Quick Health Check Commands

```bash
# Check Cloud Run Job status
gcloud run jobs describe trading-agent-job --region=us-central1

# View recent logs
gcloud run jobs executions logs <execution-id>

# Check Firestore for today's stats
# (via Google Cloud Console)
```

---

## Success Criteria

✅ 10-15 quality alerts during trading hours  
✅ No duplicate/spam alerts  
✅ EOD summary with non-zero event counts  
✅ No messages after market close  
✅ All signal types tracked correctly

---

**Status**: All changes deployed and ready for Dec 5 testing! 🚀
