# Option Chain Integration - Implementation Plan

## Goal
Add option chain analysis to support 20-30 point scalping with direction confirmation and conflict resolution.

---

## Option Chain Metrics to Track

### 1. Put-Call Ratio (PCR)
**Formula**: `Total Put OI / Total Call OI`

**Interpretation**:
- PCR > 1.2 → **Bullish** (heavy put writing = expecting up move)
- PCR 0.8-1.2 → **Neutral**
- PCR < 0.8 → **Bearish** (heavy call writing = expecting down move)

**Use Case**: Direction confirmation when spot analysis conflicts

---

### 2. Max Pain
**Definition**: Strike price where most options expire worthless

**Calculation**:
```python
# For each strike:
call_pain = sum((strike - lower_strikes) * call_OI for each lower strike)
put_pain = sum((higher_strikes - strike) * put_OI for each higher strike)
total_pain = call_pain + put_pain

# Max pain = strike with minimum total_pain
```

**Interpretation**:
- Price gravitates toward max pain (especially on expiry)
- Acts as magnet/support/resistance

**Use Case**: Adjust TP targets, avoid trades away from max pain

---

### 3. Open Interest (OI) Analysis
**Key Strikes**:
- Highest Call OI → Strong resistance
- Highest Put OI → Strong support

**OI Changes**:
- OI buildup with price rise → Bullish continuation
- OI unwinding with price rise → Bearish reversal imminent

**Use Case**: Identify key support/resistance levels beyond technical S/R

---

### 4. Implied Volatility (IV)
**ATM IV spike** → Expecting big move (good for scalping)
**IV crush** → Low expected movement (avoid trading)

**Use Case**: Filter out low-volatility periods

---

## Data Source: NSE Official API

### Endpoint
```
https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY
https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY
```

### Rate Limits
- ~2-3 requests per minute
- Need proper headers to avoid blocks

### Sample Response Structure
```json
{
  "records": {
    "expiryDates": ["05-Dec-2024", "12-Dec-2024"],
    "data": [
      {
        "strikePrice": 26000,
        "expiryDate": "05-Dec-2024",
        "CE": {
          "openInterest": 45000,
          "changeinOpenInterest": 2000,
          "impliedVolatility": 12.5,
          "lastPrice": 120
        },
        "PE": {
          "openInterest": 55000,
          "changeinOpenInterest": -1000,
          "impliedVolatility": 13.2,
          "lastPrice": 95
        }
      }
    ]
  }
}
```

---

## Implementation Plan

### Phase 1: Data Fetching (New Module)

**File**: `data_module/option_chain_fetcher.py`

```python
class OptionChainFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0...',
            'Accept': 'application/json',
            'Referer': 'https://www.nseindia.com/'
        })
        self.cache = {}
        self.cache_ttl = 60  # 1 minute
    
    def fetch_option_chain(self, instrument: str) -> Optional[Dict]:
        """Fetch option chain for NIFTY or BANKNIFTY"""
        symbol = "NIFTY" if instrument == "NIFTY" else "BANKNIFTY"
        
        # Check cache
        cache_key = f"oc_{symbol}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Cache result
            self.cache[cache_key] = data
            self.cache_time[cache_key] = time.time()
            
            return data
        except Exception as e:
            logger.error(f"Failed to fetch option chain for {symbol}: {e}")
            return None
```

---

### Phase 2: Metric Calculation (New Module)

**File**: `analysis_module/option_chain_analyzer.py`

```python
class OptionChainAnalyzer:
    def calculate_pcr(self, option_data: Dict) -> Optional[float]:
        """Calculate Put-Call Ratio"""
        total_call_oi = 0
        total_put_oi = 0
        
        for strike_data in option_data['records']['data']:
            if 'CE' in strike_data:
                total_call_oi += strike_data['CE'].get('openInterest', 0)
            if 'PE' in strike_data:
                total_put_oi += strike_data['PE'].get('openInterest', 0)
        
        if total_call_oi == 0:
            return None
        
        pcr = total_put_oi / total_call_oi
        return pcr
    
    def calculate_max_pain(self, option_data: Dict, spot_price: float) -> Optional[float]:
        """Calculate max pain strike"""
        strikes = []
        pain_values = {}
        
        for strike_data in option_data['records']['data']:
            strike = strike_data['strikePrice']
            strikes.append(strike)
            
            call_oi = strike_data.get('CE', {}).get('openInterest', 0)
            put_oi = strike_data.get('PE', {}).get('openInterest', 0)
            
            # Calculate pain at this strike
            call_pain = sum((strike - s) * data['CE']['openInterest'] 
                          for s, data in strikes_below if s < strike)
            put_pain = sum((s - strike) * data['PE']['openInterest'] 
                         for s, data in strikes_above if s > strike)
            
            pain_values[strike] = call_pain + put_pain
        
        # Max pain = strike with minimum pain
        if pain_values:
            max_pain_strike = min(pain_values, key=pain_values.get)
            return max_pain_strike
        
        return None
    
    def get_key_strikes(self, option_data: Dict, spot_price: float) -> Dict:
        """Get strikes with highest OI (support/resistance)"""
        call_oi_by_strike = {}
        put_oi_by_strike = {}
        
        for strike_data in option_data['records']['data']:
            strike = strike_data['strikePrice']
            
            if 'CE' in strike_data:
                call_oi_by_strike[strike] = strike_data['CE'].get('openInterest', 0)
            if 'PE' in strike_data:
                put_oi_by_strike[strike] = strike_data['PE'].get('openInterest', 0)
        
        # Top 3 call OI strikes (resistance)
        top_call_strikes = sorted(call_oi_by_strike.items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
        
        # Top 3 put OI strikes (support)
        top_put_strikes = sorted(put_oi_by_strike.items(), 
                                key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "resistance_strikes": [s for s, oi in top_call_strikes],
            "support_strikes": [s for s, oi in top_put_strikes],
            "max_call_oi": top_call_strikes[0] if top_call_strikes else None,
            "max_put_oi": top_put_strikes[0] if top_put_strikes else None
        }
```

---

### Phase 3: Integration with Signal Generation

**File**: `main.py` - Modify `_generate_signals()`

```python
def _generate_signals(self, instrument: str, analysis: Dict, nse_data: Dict) -> List[Dict]:
    """Generate signals with option chain confirmation"""
    
    # EXISTING filters...
    
    # NEW: Fetch option chain data
    option_chain = self.option_chain_fetcher.fetch_option_chain(instrument)
    option_metrics = {}
    
    if option_chain:
        oc_analyzer = OptionChainAnalyzer()
        option_metrics = {
            "pcr": oc_analyzer.calculate_pcr(option_chain),
            "max_pain": oc_analyzer.calculate_max_pain(option_chain, current_price),
            "key_strikes": oc_analyzer.get_key_strikes(option_chain, current_price)
        }
        logger.info(f"📊 Option Chain | PCR: {option_metrics['pcr']:.2f} | Max Pain: {option_metrics['max_pain']:.0f}")
    
    # EXISTING pattern detection...
    
    # NEW: Use option data for conflict resolution
    if len(signals) > 1:
        long_signals = [s for s in signals if "BULLISH" in s["signal_type"] or "SUPPORT" in s["signal_type"]]
        short_signals = [s for s in signals if "BEARISH" in s["signal_type"] or "RESISTANCE" in s["signal_type"]]
        
        if long_signals and short_signals and option_metrics.get("pcr"):
            pcr = option_metrics["pcr"]
            
            # Use PCR to resolve conflict
            if pcr > 1.2:
                # Bullish sentiment - take LONG
                logger.info(f"✅ PCR {pcr:.2f} confirms BULLISH - taking LONG signal")
                signals = long_signals
            elif pcr < 0.8:
                # Bearish sentiment - take SHORT
                logger.info(f"✅ PCR {pcr:.2f} confirms BEARISH - taking SHORT signal")
                signals = short_signals
            else:
                # Neutral - keep highest confidence
                logger.warning(f"⚠️ PCR {pcr:.2f} neutral - taking highest confidence")
                signals = [max(signals, key=lambda x: x.get('confidence', 0))]
    
    # NEW: Adjust confidence based on option data
    for sig in signals:
        direction = "LONG" if any(x in sig["signal_type"] for x in ["BULLISH", "SUPPORT"]) else "SHORT"
        
        if option_metrics.get("pcr"):
            pcr = option_metrics["pcr"]
            
            # Boost confidence if PCR aligns with direction
            if direction == "LONG" and pcr > 1.2:
                sig["confidence"] = min(100, sig["confidence"] + 5)
                sig["description"] += f" | PCR {pcr:.2f} confirms bullish"
            elif direction == "SHORT" and pcr < 0.8:
                sig["confidence"] = min(100, sig["confidence"] + 5)
                sig["description"] += f" | PCR {pcr:.2f} confirms bearish"
            elif (direction == "LONG" and pcr < 0.8) or (direction == "SHORT" and pcr > 1.2):
                # PCR contradicts - reduce confidence
                sig["confidence"] = max(0, sig["confidence"] - 10)
                sig["description"] += f" | ⚠️ PCR {pcr:.2f} contradicts"
    
    return signals
```

---

## Testing Plan

### 1. Unit Tests
- Test PCR calculation
- Test max pain calculation
- Test key strikes extraction

### 2. Integration Tests  
- Fetch real option chain data
- Verify conflict resolution with PCR
- Validate confidence adjustments

### 3. Live Testing
- Monitor Dec 6 session
- Compare signals with/without option data
- Track conflict resolution accuracy

---

## Expected Impact

### Conflict Resolution
**Before**: Both LONG + SHORT sent (unusable)
**After**: PCR-based direction selection (tradeable)

### Signal Quality
**Before**: 65-75% confidence average
**After**: 70-80% with option confirmation

### Target Accuracy
**Before**: TP sometimes blocked by hidden resistance
**After**: Max pain awareness prevents "into-the-wall" trades

---

## Deployment Steps

1. **Create new modules** (`option_chain_fetcher.py`, `option_chain_analyzer.py`)
2. **Test locally** with real NSE data
3. **Integrate** into `_generate_signals()`
4. **Deploy** Saturday evening
5. **Monitor** Sunday (if market) or Monday

---

## Estimated Time
- **Phase 1** (Fetcher): 1-2 hours
- **Phase 2** (Analyzer): 2-3 hours
- **Phase 3** (Integration): 1-2 hours
- **Testing**: 1 hour
- **Total**: 5-8 hours

---

## Next Steps

1. Review this plan
2. Implement Phase 1 (fetcher) first
3. Test with live NSE data
4. Build out Phases 2-3

**Ready to proceed?**
