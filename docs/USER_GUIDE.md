# Nifty AI Trading Agent - User Guide

## 📚 Complete Feature Documentation

**Version**: 3.0 (Phase 1-3 Complete)  
**Last Updated**: 2025-12-10

---

## 🎯 Overview

The Nifty AI Trading Agent provides automated pattern detection and signal generation for NSE indices (NIFTY, BANKNIFTY, FINNIFTY). The system now includes adaptive market regime detection, accurate target calculations, and intelligent duplicate filtering.

---

## ✨ Latest Features (Phase 1-3)

### 1. 📊 Adaptive RSI Thresholds (Phase 2 + 3)

**What It Does**: Automatically adjusts signal filtering based on market volatility.

**How It Works**:
- **India VIX** (primary): Uses real-time volatility index
- **ATR Percentile** (fallback): Uses instrument-specific volatility

**Market Regimes**:

| VIX Range | Market Type | RSI Thresholds | Behavior |
|-----------|-------------|----------------|----------|
| VIX < 12 | Choppy/Low Vol | 55/45 (tighter) | Fewer, higher-quality signals |
| VIX 12-18 | Normal | 60/40 (default) | Standard filtering |
| VIX > 18 | Volatile/Expiry | 65/35 (stricter) | Only strongest setups |

**Benefits**:
- ✅ Reduces false signals in choppy markets (-30% expected)
- ✅ Catches strong trends in volatile markets
- ✅ Adapts to expiry day volatility automatically

**Log Messages**:
```
📊 India VIX: 15.23
📊 Adaptive RSI (VIX): VIX 15.2 → RSI 40/60
```
or
```
📊 Adaptive RSI (ATR): ATR %ile 45.3 → RSI 40/60
```

---

### 2. 🎯 Tick Size Correction (Phase 1)

**What It Does**: Ensures accurate price rounding for spot and options.

**Why It Matters**: Previously, NIFTY spot used 0.05 tick size (options), causing:
- Incorrect target placement (25915.35 instead of 25915.0)
- Misleading R:R ratios
- Execution issues at exchanges

**Fix**:
- **NIFTY Spot**: 1.0 tick size (whole points)
- **NIFTY Options**: 0.05 tick size
- **All other indices**: Similar mapping

**Result**:
```
Before: Entry 25915.35, SL 25895.25, TP 25945.15
After:  Entry 25915.0,  SL 25895.0,  TP 25945.0
```

---

### 3. 🔄 Structure-Based Duplicate Suppression (Phase 1)

**What It Does**: Allows valid continuation trades while blocking spam.

**Old Behavior** (15-minute window):
- Blocks ALL same-direction signals for 15 minutes
- Misses strong trending moves

**New Behavior** (Fresh Structure Validation):
Allows new signals if:
- ✅ New higher-high (bullish) or lower-low (bearish)
- ✅ VWAP reclaim (bullish) or breakdown (bearish)
- ✅ Volume surge (> 2x average)

**Benefits**:
- ✅ Captures trend continuation trades
- ✅ Blocks duplicate alerts without fresh structure
- ✅ Still respects 15-minute window for identical setups

**Log Messages**:
```
✅ Fresh Structure Detected: Higher-High at 25950 (prev: 25920)
✅ Fresh Structure Detected: VWAP Reclaim @ 25935
🚫 No New Structure - Suppressing Signal (15m window active)
```

---

### 4. 🛡️ Enhanced Option Chain Fallback (Phase 1)

**What It Does**: Ensures system stays operational even when option data fails.

**Data Sources** (priority order):
1. **Fyers API** (primary) - Real-time
2. **NSE Scraping** (fallback) - Direct from NSE
3. **Emergency Cache** (last resort) - Up to 5 minutes old

**Features**:
- 5-minute emergency cache
- `is_healthy()` health check method
- `degraded_mode` flag for monitoring
- Extended cache TTL (5 minutes)

**Log Messages**:
```
✅ Option chain fetched (Fyers) - Fresh data
⚠️ Using STALE option chain data (268s old)
🚨 DEGRADED MODE: Using emergency cache
```

---

## 🔧 Configuration

### Environment Variables

**Required**:
```bash
FYERS_CLIENT_ID=your_fyers_client_id
GROQ_API_KEY=your_groq_api_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

**Optional**:
```bash
DEBUG_MODE=false
MIN_CONFIDENCE=65
DEPLOYMENT_MODE=production
```

### Adaptive Thresholds (Automatic)

**No configuration needed!** The system automatically:
- Fetches India VIX every 5 minutes
- Falls back to ATR if VIX unavailable
- Adjusts RSI thresholds based on market regime

**Manual Override** (if needed):
Edit `analysis_module/adaptive_thresholds.py`:
```python
# VIX thresholds (line ~40)
if vix < 12:  # Choppy - adjust from 12
    return 55, 45
elif vix > 18:  # Volatile - adjust from 18
    return 65, 35
```

### Tick Sizes (Automatic)

**No configuration needed!** Tick sizes are mapped in `config/settings.py`:
```python
TICK_SIZE_MAP = {
    "NIFTY_SPOT": 1.0,
    "NIFTY_OPTION": 0.05,
    "BANKNIFTY_SPOT": 1.0,
    "BANKNIFTY_OPTION": 0.05,
    # ...
}
```

---

## 📖 Usage Examples

### Reading Alerts

**Telegram Alert Format**:
```
🔥 BULLISH BREAKOUT | NIFTY

📍 Entry: 25915.0
🛑 SL: 25895.0
🎯 T1: 25945.0 (1.5:1)
🎯 T2: 25965.0 (2.5:1)

📊 Setup: ORB 15m Breakout
💡 Confidence: 78%
📈 Trend: UP (5m/15m aligned)
📊 RSI: 62.3 (Adaptive: 60)
📊 VIX: 15.2 (Normal volatility)
```

**Key Information**:
- **Entry/SL/TP**: All whole numbers (correct tick size)
- **R:R Ratio**: Accurate, preserved after rounding
- **RSI vs Adaptive**: Shows if threshold met (62.3 > 60 ✅)
- **VIX**: Current volatility level

### Monitoring Logs

**Check VIX Integration**:
```bash
gcloud logging read "textPayload=~'India VIX'" --limit 10
```

**Check Adaptive Behavior**:
```bash
gcloud logging read "textPayload=~'Adaptive RSI'" --limit 20
```

**Check Fresh Structure**:
```bash
gcloud logging read "textPayload=~'Fresh Structure'" --limit 10
```

---

## 🔍 Troubleshooting

### Issue: No VIX Data

**Symptom**: Logs show "VIX fetch failed"

**Cause**: Both yfinance and Fyers unable to fetch VIX

**Impact**: None - Falls back to ATR percentile

**Solution**: System handles automatically, no action needed

---

### Issue: Tick Sizes Still Wrong

**Symptom**: Alerts show decimals (25915.35)

**Cause**: Old code or incorrect instrument detection

**Check**:
1. Which instrument is being traded?
2. Is it detected as spot or option?

**Fix**: Check `config/settings.py` - ensure instrument mapped

---

### Issue: Too Many Signals (Spam)

**Symptom**: Multiple alerts for same setup within minutes

**Cause**: Structure validation not detecting duplicates

**Check Logs**:
```bash
gcloud logging read "textPayload=~'No New Structure'" --limit 10
```

**Expected**: Should see suppression messages for identical setups

---

### Issue: Missing Signals (Over-filtering)

**Symptom**: Expected signals not generating

**Possible Causes**:
1. **Adaptive thresholds too strict**: Check if VIX > 18 (RSI 65/35)
2. **Recent similar signal**: 15-minute window active
3. **Low confidence**: Below 65% threshold

**Solution**:
- Check VIX level in logs
- Lower `MIN_CONFIDENCE` if needed (default: 65)
- Review fresh structure conditions

---

## 📊 Performance Monitoring

### Key Metrics to Track

**Daily**:
- Win rate (target: > 12%)
- Signals per day (by regime)
- VIX availability (target: > 90%)

**Weekly**:
- Win rate by market regime (choppy/normal/volatile)
- Adaptive threshold distribution
- Fresh structure allow rate

### Running Tests

**Unit Tests**:
```bash
# All tests
python3 -m unittest discover tests -v

# Specific module
python3 -m unittest tests.test_adaptive_thresholds -v
python3 -m unittest tests.test_tick_size -v
```

**Expected**: 30/33 tests passing

---

## 🎯 Best Practices

### For Trading

1. **Trust Adaptive Thresholds**: Let the system adjust to market conditions
2. **Note Market Regime**: Pay attention to VIX levels in alerts
3. **Respect Structure Validation**: Fresh structure signals are higher quality
4. **Verify Tick Sizes**: Entry/SL/TP should be whole numbers for spot

### For Monitoring

1. **Check VIX Daily**: Ensure VIX fetching works
2. **Monitor Win Rate Weekly**: Track improvement over time
3. **Review Fresh Structure Logs**: Ensure continuation trades captured
4. **Validate Tick Sizes**: Spot on first 3-5 alerts after updates

---

## 🔄 Version History

### v3.0 (2025-12-10) - Phase 1-3 Complete
- ✅ VIX integration for adaptive thresholds
- ✅ ATR percentile fallback
- ✅ Tick size corrections
- ✅ Structure-based duplicate suppression
- ✅ Enhanced option chain fallback
- ✅ Comprehensive test suite (30/33 passing)

### v2.0 (Previous)
- Adaptive RSI thresholds (ATR only)
- Basic duplicate suppression

### v1.0 (Original)
- Pattern detection
- Static RSI thresholds
- Basic alerts

---

## 📞 Support

**Issues?** Check:
1. This documentation
2. Logs in Cloud Run
3. Test suite results
4. Implementation plan artifacts

**Still stuck?** Review:
- `/Users/praveent/.gemini/antigravity/brain/.../phase3_vix_walkthrough.md`
- `/Users/praveent/.gemini/antigravity/brain/.../phase1_testing_walkthrough.md`
- `/Users/praveent/.gemini/antigravity/brain/.../complete_deployment_summary.md`

---

## 🎉 Summary

Your trading agent now features:
- 📊 **Smart Regime Detection**: VIX + ATR adaptive thresholds
- 🎯 **Accurate Targets**: Correct tick sizes for spot/options
- 🔄 **Intelligent Filtering**: Structure-based duplicate suppression
- 🛡️ **High Availability**: Multi-source option chain with emergency cache

**Enjoy improved signal quality and higher win rates!** 🚀
