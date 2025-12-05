# EUR/USD London Session Trading Agent - Implementation Plan

## Goal
Build automated EUR/USD trading alert system for London session (12:30-3:00 PM IST) using breakout-retest strategy, targeting 10-15 pip moves with ₹1000-1500 daily profit.

---

## Strategy Summary

**Session**: 12:30-3:00 PM IST  
**Instruments**: EUR/USD  
**Timeframes**: 15m structure, 5m entry  
**Opening Range**: Asian Session High/Low (5:30 AM - 11:30 AM IST)  
**Setups**:
1. Breakout → Retest → Continuation
2. Liquidity Grab → Reversal → Retest

**Target**: 10-15 pips per trade (1-2 trades/day)  
**Risk**: 8-12 pips SL, ₹300-500 per trade

---

## Free Forex Data APIs (Real-time)

### Option 1: Twelve Data ⭐ RECOMMENDED
- **URL**: https://twelvedata.com/
- **Free tier**: 800 API calls/day
- **Update**: Real-time (WebSocket available)
- **Coverage**: EUR/USD with 1m, 5m, 15m, 1h, 1d data
- **Pros**: Free, reliable, good documentation
- **Cons**: Rate limit (enough for 5-min polling)

### Option 2: Alpha Vantage
- **URL**: https://www.alphavantage.co/
- **Free tier**: 25 calls/day (too limited)
- **Not suitable for 5-min execution**

### Option 3: Forex API (fcsapi.com)
- **URL**: https://fcsapi.com/
- **Free tier**: 500 calls/month
- **Not suitable**

**Decision**: Use **Twelve Data** with 5-minute polling

---

## Architecture

### Reuse from NIFTY Agent
```
nifty-ai-trading-agent/
├── analysis_module/
│   └── technical.py       ✅ Reuse (adapt for pips)
├── telegram_module/
│   └── bot_handler.py     ✅ Reuse (minor changes)
├── data_module/
│   ├── persistence.py     ✅ Reuse
│   └── fetcher.py         🔄 Replace with forex_fetcher.py
├── config/
│   └── settings.py        🔄 New forex_settings.py
└── main.py                🔄 New eurusd_main.py
```

### New Components
```
eurusd-london-agent/
├── data_module/
│   └── forex_fetcher.py   🆕 Twelve Data integration
├── analysis_module/
│   ├── asian_range.py     🆕 Asian session tracker
│   └── news_filter.py     🆕 Economic calendar
├── config/
│   └── forex_settings.py  🆕 EUR/USD specific config
└── eurusd_main.py         🆕 Main orchestrator
```

---

## Phase 1: Core Infrastructure (2-3 hours)

### 1.1 Project Setup
```bash
# Create new directory
mkdir -p /Users/praveent/eurusd-london-agent
cd /Users/praveent/eurusd-london-agent

# Copy reusable modules
cp -r /Users/praveent/nifty-ai-trading-agent/telegram_module .
cp -r /Users/praveent/nifty-ai-trading-agent/data_module/persistence.py data_module/
```

### 1.2 Forex Data Fetcher
**File**: `data_module/forex_fetcher.py`

```python
import requests
import pandas as pd
from datetime import datetime, timedelta

class ForexDataFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
    
    def fetch_ohlc(self, symbol: str, interval: str, outputsize: int = 100):
        """
        Fetch OHLC data for forex pair.
        
        Args:
            symbol: "EUR/USD"
            interval: "1min", "5min", "15min"
            outputsize: Number of candles
        """
        url = f"{self.base_url}/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
            "format": "JSON"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        values = data.get("values", [])
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        
        # Convert to float
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        
        return df.sort_index()
```

### 1.3 Asian Range Calculator
**File**: `analysis_module/asian_range.py`

```python
def calculate_asian_range(df: pd.DataFrame) -> Dict:
    """
    Calculate Asian session high/low (5:30 AM - 11:30 AM IST).
    
    Returns:
        {
            "high": float,
            "low": float,
            "range_pips": float
        }
    """
    # Filter to Asian session (convert UTC to IST as needed)
    asian_session = df.between_time("05:30", "11:30")
    
    if asian_session.empty:
        return None
    
    asian_high = asian_session["high"].max()
    asian_low = asian_session["low"].min()
    range_pips = (asian_high - asian_low) * 10000  # Convert to pips
    
    return {
        "high": asian_high,
        "low": asian_low,
        "range_pips": range_pips
    }
```

---

## Phase 2: Signal Detection (3-4 hours)

### 2.1 Breakout Detection (Adapt from NIFTY)
**File**: `analysis_module/forex_technical.py`

Key changes:
- Use **pips** instead of points (1 pip = 0.0001 for EUR/USD)
- Asian range as opening range (not PDH/PDL)
- 8-12 pip SL, 10-15 pip TP

```python
def detect_breakout(df_5m, df_15m, asian_range):
    """
    Detect breakout → retest setup.
    
    Criteria:
    1. 15m candle closes outside Asian range
    2. Wait for retest on 5m
    3. Confirmation: engulfing, rejection wick, or structure shift
    """
    current_price = df_5m.iloc[-1]["close"]
    
    # Bullish breakout
    if current_price > asian_range["high"]:
        # Wait for retest
        if df_5m.iloc[-2]["low"] <= asian_range["high"]:
            # Check confirmation
            if is_bullish_confirmation(df_5m.iloc[-1]):
                return {
                    "type": "BREAKOUT_RETEST",
                    "direction": "LONG",
                    "entry": current_price,
                    "sl": current_price - 0.0012,  # 12 pips
                    "tp": current_price + 0.0015,  # 15 pips
                    "level": asian_range["high"]
                }
    
    # Bearish breakout (similar logic)
    ...
```

### 2.2 Liquidity Grab Detection
```python
def detect_liquidity_grab(df_5m, asian_range):
    """
    Detect liquidity grab → reversal.
    
    Criteria:
    1. Wick above/below Asian range without close
    2. Reversal candle
    3. Retest of swept level
    """
    prev_candle = df_5m.iloc[-2]
    current_candle = df_5m.iloc[-1]
    
    # Bullish liquidity grab (wick below Asian low)
    if (prev_candle["low"] < asian_range["low"] and 
        prev_candle["close"] > asian_range["low"]):
        # Reversal confirmation
        if current_candle["close"] > current_candle["open"]:
            return {
                "type": "LIQUIDITY_GRAB",
                "direction": "LONG",
                ...
            }
```

---

## Phase 3: News Filter (1-2 hours)

### News Calendar API
Use **Forex Factory Calendar** or **Investing.com API**

```python
def check_news_risk(time_ist: datetime) -> bool:
    """
    Check if high-impact EUR/USD news within next 30 minutes.
    
    Avoid trading during:
    - 1:30 PM IST (8:00 AM UTC)
    - 2:30 PM IST (9:00 AM UTC)
    """
    hour = time_ist.hour
    minute = time_ist.minute
    
    # High-impact news windows
    if (hour == 13 and minute >= 20) or (hour == 14 and minute <= 0):
        return True  # Skip trading
    if (hour == 14 and minute >= 20) or (hour == 15 and minute <= 0):
        return True  # Skip trading
    
    return False
```

---

## Phase 4: Main Orchestrator (2 hours)

### Session Logic
```python
def is_london_session():
    """Check if current time is London session (12:30-3:00 PM IST)"""
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    current_time = now.time()
    
    return time(12, 30) <= current_time <= time(15, 0)

def main():
    # 12:00-12:25 PM: Mark Asian range
    if time(12, 0) <= now.time() <= time(12, 25):
        asian_range = calculate_asian_range(df_5m)
        send_market_context(asian_range)
    
    # 12:30-3:00 PM: Trade
    if is_london_session():
        if not check_news_risk(now):
            signals = detect_signals(df_5m, df_15m, asian_range)
            for signal in signals:
                send_alert(signal)
```

---

## Phase 5: Deployment (1 hour)

### Cloud Run Job
```yaml
schedule: "*/5 12-15 * * 1-5"  # Every 5 min, 12:30-3:00 PM IST, Mon-Fri
timezone: "Asia/Kolkata"
```

### Environment Variables
```
TWELVE_DATA_API_KEY=<your-key>
TELEGRAM_BOT_TOKEN=<token>
TELEGRAM_CHAT_ID=<chat-id>
```

---

## Testing Plan

### 1. Backtest (Historical Data)
- Download 1 month of EUR/USD 5m data from Twelve Data
- Simulate Asian range + London session
- Expected: 30-60 signals (1-2 per day)

### 2. Paper Trading (1 week)
- Run in production with alerts only
- Verify Asian range calculation
- Check news filter effectiveness

### 3. Live Trading
- Start with 0.02 lot
- Increase to 0.05-0.10 after consistent results

---

## File Structure

```
eurusd-london-agent/
├── data_module/
│   ├── __init__.py
│   ├── forex_fetcher.py       # Twelve Data integration
│   └── persistence.py          # Firestore (reused)
├── analysis_module/
│   ├── __init__.py
│   ├── forex_technical.py      # Breakout/retest detection
│   ├── asian_range.py          # Asian session tracker
│   └── news_filter.py          # Economic calendar
├── telegram_module/
│   ├── __init__.py
│   └── bot_handler.py          # Reused from NIFTY
├── config/
│   ├── __init__.py
│   └── forex_settings.py       # EUR/USD config
├── eurusd_main.py              # Main orchestrator
├── backtest_eurusd.py          # Backtesting script
├── requirements.txt
├── Dockerfile
├── deploy_job.sh
└── README.md
```

---

## Estimated Timeline

| Phase | Task | Time | Status |
|-------|------|------|--------|
| 1 | Core infrastructure | 2-3h | ⏳ |
| 2 | Signal detection | 3-4h | ⏳ |
| 3 | News filter | 1-2h | ⏳ |
| 4 | Main orchestrator | 2h | ⏳ |
| 5 | Deployment | 1h | ⏳ |
| 6 | Testing | 2h | ⏳ |
| **Total** | | **11-14h** | |

---

## Next Steps

1. ✅ Get Twelve Data API key (free tier)
2. ✅ Create project structure
3. ✅ Implement forex data fetcher
4. ✅ Adapt technical analysis for pips
5. ✅ Add news filter
6. ✅ Deploy and test

**Ready to start?** Let me know and I'll begin with Phase 1!
