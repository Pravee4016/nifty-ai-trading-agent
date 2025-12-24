
import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import pytz
import yfinance as yf
import numpy as np
from google.cloud import firestore

# Add project root to path
sys.path.append(os.getcwd())

from ml_module.feature_extractor import extract_features
from config.settings import INSTRUMENTS

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_technical_indicators(df):
    """Calculate technical indicators needed for ML features"""
    df = df.copy()
    
    # 1. EMAs
    df['ema9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # 2. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_percent'] = (df['atr'] / df['Close']) * 100
    
    # 4. VWAP (Approximate for intraday)
    df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    # 5. Trend (Simple EMA slope approximation)
    df['trend_5m'] = np.where(df['ema20'] > df['ema50'], "UP", "DOWN")
    
    return df

def get_market_data():
    """Fetch 60 days of 5m data for NIFTY"""
    logger.info("Fetching NIFTY data...")
    ticker = "^NSEI"  # NIFTY 50
    
    # Fetch 5m data (last 60 days is max for 5m)
    df = yf.download(ticker, period="60d", interval="5m", progress=False)
    
    if df.empty:
        logger.error("Failed to fetch data")
        return None
        
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Normalize Timezone to Naive Asia/Kolkata
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    
    df = calculate_technical_indicators(df)
    return df

def backfill_data():
    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "nifty-trading-agent")
        db = firestore.Client(project=project_id)
        
        # 1. Fetch Trades
        trades_ref = db.collection("trades")
        trades = list(trades_ref.stream())
        logger.info(f"Found {len(trades)} trades")
        
        if not trades:
            logger.warning("No trades found to backfill")
            return

        # 2. Fetch Market Data
        market_data = get_market_data()
        if market_data is None:
            return

        success_count = 0
        
        # 3. Process Trades
        for trade_doc in trades:
            trade = trade_doc.to_dict()
            
            # Skip if outcome unknown
            if not trade.get("outcome") or "status" not in trade.get("outcome", {}):
                continue
                
            outcome_status = trade["outcome"]["status"]
            if outcome_status not in ["WIN", "LOSS"]:
                continue
                
            # Parse timestamp
            ts_str = trade.get("timestamp") or trade.get("date")
            try:
                trade_time = pd.to_datetime(ts_str).tz_convert("Asia/Kolkata").tz_localize(None)
            except:
                # Handle simpler formats
                trade_time = pd.to_datetime(ts_str).tz_localize(None)
            
            # Find matching candle
            # Use 'asof' to find closest previous candle
            idx = market_data.index.get_indexer([trade_time], method='pad')[0]
            if idx == -1:
                logger.warning(f"No data for {trade_time}")
                continue
                
            candle = market_data.iloc[idx]
            
            # 4. Construct Context
            htf_context = {
                "trend_5m": candle["trend_5m"],
                "trend_15m": candle["trend_5m"], # Approx
                "trend_daily": "UP", # Default
                "vwap_5m": candle["vwap"],
                "ema20": candle["ema20"],
                "ema50": candle["ema50"],
                "atr_percent": candle["atr_percent"],
                "india_vix": 13.0 # Default benign VIX
            }
            
            technical_context = {"higher_tf_context": htf_context}
            
            # 5. Extract Features
            features = extract_features(
                signal=trade,
                technical_context=technical_context,
                option_metrics={"pcr": 1.0, "iv": 13.0}, # Defaults
                market_status={"is_choppy": False}
            )
            
            # 6. Create Record
            record = {
                "signal_id": trade.get("trade_id"),
                "instrument": trade.get("instrument"),
                "signal_type": trade.get("signal_type"),
                "features": features,
                "label": 1 if outcome_status == "WIN" else 0,
                "outcome": outcome_status,
                "timestamp": datetime.now(),
                "original_timestamp": ts_str
            }
            
            # 7. Save to Firestore
            db.collection("ml_training_data").document(trade.get("trade_id")).set(record)
            success_count += 1
            
        logger.info(f"✅ Successfully backfilled {success_count} records")
        
    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)

if __name__ == "__main__":
    backfill_data()
