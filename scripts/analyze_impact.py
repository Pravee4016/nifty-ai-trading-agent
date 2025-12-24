
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import pytz
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ImpactAnalysis")

# Initialize Firestore
if not firebase_admin._apps:
    cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred, {
        "projectId": "nifty-trading-agent",
    })

db = firestore.client()

def get_market_data():
    """Fetch today's NIFTY data"""
    logger.info("Fetching NIFTY data...")
    # Get just today's data (1d interval 1m or 5m)
    # Actually, we need 5m data for the proxy logic
    df = yf.download("^NSEI", period="5d", interval="5m", progress=False)
    
    # Flatten cols
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Calc Indicators
    df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    return df

def analyze_impact():
    logger.info("🚀 Starting Impact Analysis...")
    
    # 1. Fetch Today's Signals
    today = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y%m%d")
    # Actually just look for signals from today's date
    start_of_day = datetime.now(pytz.timezone("Asia/Kolkata")).replace(hour=0, minute=0, second=0, microsecond=0)
    
    signals = []
    
    # Check 'tracked_signals_nifty'
    docs = db.collection("tracked_signals_nifty").stream()
    for doc in docs:
        d = doc.to_dict()
        ts = d.get("timestamp")
        # Handle timestamp string or datetime
        if isinstance(ts, str):
            try:
                ts = pd.to_datetime(ts).tz_convert("Asia/Kolkata")
            except:
                continue # Skip if cant parse
        
        if ts and ts > start_of_day:
            signals.append(d)
            
    # Check 'trades' (completed signals)
    docs = db.collection("trades").stream()
    for doc in docs:
        d = doc.to_dict()
        ts = d.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = pd.to_datetime(ts)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC").tz_convert("Asia/Kolkata")
            except:
                continue
        
        if ts and ts > start_of_day:
            signals.append(d)
    
    logger.info(f"📊 Found {len(signals)} signals from today.")
    
    if len(signals) == 0:
        print("No signals found for today to analyze.")
        return

    # 2. Get Data
    df = get_market_data()
    # Normalize index
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    
    # 3. Re-evaluate
    passed = 0
    filtered = 0
    details = []
    
    for s in signals:
        ts = s.get("timestamp")
        if isinstance(ts, str):
            try:
                sig_time = pd.to_datetime(ts).tz_localize(None) # Make naive
            except:
                 try:
                    sig_time = pd.to_datetime(ts).tz_convert("Asia/Kolkata").tz_localize(None)
                 except: continue
        else:
             sig_time = ts.replace(tzinfo=None)

        # Find closest candle
        idx = df.index.get_indexer([sig_time], method='pad')[0]
        if idx == -1:
            print(f"Skipping {s['signal_type']} at {sig_time}: No data")
            continue
            
        # Get context (last 20 candles up to this one)
        # Actually calculate_volume_proxy uses the *last* row of the DF passed
        # So we slice df up to idx
        df_slice = df.iloc[:idx+1]
        
        # --- LOGIC DUPLICATION FROM TECHNICAL.PY ---
        score = 0
        breakdown = []
        
        # 1. VWAP (+2)
        # VP_ATR_MULT = 0.5, VP_VWAP_LOOKBACK = 5
        sl_slice = df_slice.iloc[-1]
        vwap = sl_slice["vwap"]
        atr = sl_slice["atr"]
        close = sl_slice["Close"]
        open_ = sl_slice["Open"]
        
        # Use Analyzer Logic directly to verify the fix
        from analysis_module.technical import TechnicalAnalyzer
        
        # Instantiate once globally or here? Here is fine
        analyzer = TechnicalAnalyzer("NIFTY")
        
        # Pass the slice to the function
        # Note: calculate_volume_proxy expects the full DF context usually, pass slice
        result = analyzer.calculate_volume_proxy(df_slice, option_chain_data=None)
        
        score = result["score"]
        breakdown = result["breakdown"]
        
        # Determine status based on new relaxed logic
        # Settings imported inside the method will pick up env vars or defaults
        # We need to reload settings if they changed?
        # Since script is fresh run, it picks up new settings.py values (3)
        
        from config.settings import VP_MIN_SCORE
        
        threshold = VP_MIN_SCORE # Should be 3 now
        
        # Max potential score logic
        # If Options passed (+2)
        score_diff = 2 # Assuming options were perfect
        max_pot = score + score_diff
        
        status = "UNKNOWN"
        if score >= threshold:
             status = "PASSED" # Passed even without options!
             passed += 1
        elif max_pot < threshold:
            status = "FILTERED"
            filtered += 1
        else:
             status = "DEPENDENT" # Would pass if Options were good
             passed += 1 

        details.append({
            "time": sig_time.strftime("%H:%M"),
            "type": s.get("signal_type"),
            "tech_score": score,
            "breakdown": breakdown,
            "status": status
        })

    # Report
    from config.settings import VP_MIN_SCORE
    print(f"\n📢 IMPACT ANALYSIS FOR TODAY ({len(details)} Signals)")
    print(f"Logic: Updated TechnicalAnalyzer | Threshold: {VP_MIN_SCORE}")
    print("-" * 80)
    print(f"{'TIME':<10} {'TYPE':<20} {'SCORE':<8} {'STATUS':<12} {'DETAILS'}")
    print("-" * 80)
    
    for d in details:
        print(f"{d['time']:<10} {d['type']:<20} {d['tech_score']:<8} {d['status']:<12} {d['breakdown']}")
        
    print("-" * 60)
    print(f"📉 Filtered (Guaranteed): {filtered}")
    print(f"✅ Potential Pass: {passed}")
    print(f"NOTE: 'Potential Pass' assumes Options data was favorable (+2).")
    print(f"Signals with Tech Score < 2 (VWAP failed) are guaranteed filtered irrespective of options.")

if __name__ == "__main__":
    analyze_impact()
