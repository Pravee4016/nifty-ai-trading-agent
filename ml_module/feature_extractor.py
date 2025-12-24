"""
Feature Extraction for LightGBM Model
Converts signal + context data into ML features
"""

import logging
from typing import Dict
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)


def extract_features(
    signal: Dict,
    technical_context: Dict,
    option_metrics: Dict,
    market_status: Dict = None
) -> Dict:
    """
    Extract 30+ features from signal data for ML prediction.
    
    Args:
        signal: Trading signal with type, entry, SL, TP, etc.
        technical_context: MTF analysis, trends, indicators
        option_metrics: PCR, IV, OI data
        market_status: Optional choppy/volatile conditions
        
    Returns:
        Dict of features for LightGBM
    """
    try:
        # Initialize features dict
        features = {}
        
        # =================================================================
        # 1. SIGNAL CHARACTERISTICS
        # =================================================================
        features["signal_type"] = signal.get("signal_type", "UNKNOWN")  # Categorical
        features["confidence"] = float(signal.get("confidence", 0))
        features["volume_confirmed"] = int(signal.get("volume_confirmed", False))
        features["momentum_confirmed"] = int(signal.get("momentum_confirmed", False))
        features["risk_reward"] = float(signal.get("risk_reward_ratio", 0))
        
        # Price levels
        entry = float(signal.get("entry_price", 0))
        sl = float(signal.get("stop_loss", 0))
        tp = float(signal.get("take_profit", 0))
        
        features["stop_loss_pct"] = abs(entry - sl) / entry * 100 if entry > 0 else 0
        features["target_pct"] = abs(tp - entry) / entry * 100 if entry > 0 else 0
        
        # =================================================================
        # 2. MULTI-TIMEFRAME CONTEXT
        # =================================================================
        htf_context = technical_context.get("higher_tf_context", {})
        
        features["trend_5m"] = htf_context.get("trend_5m", "NEUTRAL")  # Categorical
        features["trend_15m"] = htf_context.get("trend_15m", "NEUTRAL")
        features["trend_daily"] = htf_context.get("trend_daily", "NEUTRAL")
        
        # Trend alignment (binary)
        signal_direction = "UP" if "BULLISH" in features["signal_type"] or "SUPPORT" in features["signal_type"] else "DOWN"
        features["trend_aligned_15m"] = int(
            (signal_direction == "UP" and features["trend_15m"] == "UP") or
            (signal_direction == "DOWN" and features["trend_15m"] == "DOWN")
        )
        features["trend_aligned_daily"] = int(
            (signal_direction == "UP" and features["trend_daily"] == "UP") or
            (signal_direction == "DOWN" and features["trend_daily"] == "DOWN")
        )
        
        # =================================================================
        # 3. PRICE STRUCTURE INDICATORS
        # =================================================================
        vwap = htf_context.get("vwap_5m", entry)
        ema20 = htf_context.get("ema20", entry)
        ema50 = htf_context.get("ema50", entry)
        
        features["distance_to_vwap_pct"] = (entry - vwap) / vwap * 100 if vwap > 0 else 0
        features["distance_to_ema20_pct"] = (entry - ema20) / ema20 * 100 if ema20 > 0 else 0
        features["distance_to_ema50_pct"] = (entry - ema50) / ema50 * 100 if ema50 > 0 else 0
        
        features["above_vwap"] = int(entry > vwap)
        features["above_ema20"] = int(entry > ema20)
        
        # ATR-based volatility
        features["atr_percent"] = float(htf_context.get("atr_percent", 0))
        
        # =================================================================
        # 4. OPTIONS DATA
        # =================================================================
        features["pcr"] = float(option_metrics.get("pcr", 1.0))
        features["iv"] = float(option_metrics.get("iv", 15))
        
        oi_data = option_metrics.get("oi_change", {})
        features["oi_sentiment"] = oi_data.get("sentiment", "NEUTRAL")  # Categorical
        
        # Option alignment with signal
        oi_bullish = features["oi_sentiment"] == "BULLISH"
        oi_bearish = features["oi_sentiment"] == "BEARISH"
        features["oi_aligned"] = int(
            (signal_direction == "UP" and oi_bullish) or
            (signal_direction == "DOWN" and oi_bearish)
        )
        
        # =================================================================
        # 5. TIME-BASED FEATURES
        # =================================================================
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)
        
        features["hour_of_day"] = now.hour
        features["minute_of_hour"] = now.minute
        features["day_of_week"] = now.weekday()  # 0=Monday, 4=Friday
        
        # Market session
        market_open = datetime.strptime("09:15", "%H:%M").time()
        current_time = now.time()
        minutes_from_open = (now.hour * 60 + now.minute) - (9 * 60 + 15)
        
        features["minutes_from_open"] = max(0, minutes_from_open)
        features["is_first_hour"] = int(minutes_from_open < 60)
        features["is_last_hour"] = int(now.hour >= 14 and now.minute >= 30)
        features["is_lunch_hour"] = int(12 <= now.hour < 13)
        
        # =================================================================
        # 6. MARKET CONDITIONS
        # =================================================================
        features["india_vix"] = float(htf_context.get("india_vix", 15))
        features["vix_regime"] = "HIGH" if features["india_vix"] > 20 else "NORMAL"  # Categorical
        
        if market_status:
            features["is_choppy"] = int(market_status.get("is_choppy", False))
        else:
            features["is_choppy"] = 0
        
        # =================================================================
        # 7. PATTERN-SPECIFIC FEATURES
        # =================================================================
        is_breakout = "BREAKOUT" in features["signal_type"]
        is_retest = "RETEST" in features["signal_type"] or "BOUNCE" in features["signal_type"]
        is_reversal = "PIN_BAR" in features["signal_type"] or "ENGULFING" in features["signal_type"]
        
        features["is_breakout"] = int(is_breakout)
        features["is_retest"] = int(is_retest)
        features["is_reversal"] = int(is_reversal)
        features["is_inside_bar"] = int("INSIDE_BAR" in features["signal_type"])
        
        logger.debug(f"Extracted {len(features)} features for {features['signal_type']}")
        
        return features
        
    except Exception as e:
        logger.error(f"❌ Feature extraction failed: {e}", exc_info=True)
        # Return minimal features on error
        return {
            "signal_type": "ERROR",
            "confidence": 0,
            "risk_reward": 0
        }


def get_feature_names() -> list:
    """Return list of all feature names for model training."""
    return [
        # Signal characteristics
        "signal_type", "confidence", "volume_confirmed", "momentum_confirmed",
        "risk_reward", "stop_loss_pct", "target_pct",
        
        # Multi-timeframe
        "trend_5m", "trend_15m", "trend_daily",
        "trend_aligned_15m", "trend_aligned_daily",
        
        # Price structure
        "distance_to_vwap_pct", "distance_to_ema20_pct", "distance_to_ema50_pct",
        "above_vwap", "above_ema20", "atr_percent",
        
        # Options
        "pcr", "iv", "oi_sentiment", "oi_aligned",
        
        # Time-based
        "hour_of_day", "minute_of_hour", "day_of_week",
        "minutes_from_open", "is_first_hour", "is_last_hour", "is_lunch_hour",
        
        # Market conditions
        "india_vix", "vix_regime", "is_choppy",
        
        # Pattern types
        "is_breakout", "is_retest", "is_reversal", "is_inside_bar"
    ]


def get_categorical_features() -> list:
    """Return list of categorical feature names."""
    return [
        "signal_type",
        "trend_5m",
        "trend_15m",
        "trend_daily",
        "oi_sentiment",
        "vix_regime"
    ]
