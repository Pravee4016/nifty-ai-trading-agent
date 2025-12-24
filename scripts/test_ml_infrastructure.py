"""
Test ML Infrastructure with Historical Data
Uses yfinance to generate synthetic training data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yfinance as yf

from ml_module.feature_extractor import extract_features, get_feature_names
from ml_module.predictor import SignalQualityPredictor
from ml_module.model_storage import ModelStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_historical_data(symbol: str = "^NSEI", days: int = 60) -> pd.DataFrame:
    """Fetch historical data from yfinance."""
    logger.info(f"Fetching {days} days of data for {symbol}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fetch data
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval="5m")
    
    if df.empty:
        logger.error("No data fetched from yfinance")
        return None
    
    logger.info(f"✅ Fetched {len(df)} candles")
    return df


def generate_synthetic_signals(df: pd.DataFrame, num_signals: int = 100) -> list:
    """
    Generate synthetic trading signals from historical data.
    
    Simulates breakout/retest signals with realistic features.
    """
    logger.info(f"Generating {num_signals} synthetic signals...")
    
    signals = []
    ist = pytz.timezone("Asia/Kolkata")
    
    # Sample random candles (avoid first/last 20 candles)
    indices = np.random.choice(
        range(20, len(df) - 20),
        size=min(num_signals, len(df) - 40),
        replace=False
    )
    
    for idx in indices:
        candle = df.iloc[idx]
        prev_candles = df.iloc[max(0, idx-20):idx]
        
        # Determine signal type based on price action
        is_bullish = candle['Close'] > candle['Open']
        high_break = candle['High'] > prev_candles['High'].max()
        low_break = candle['Low'] < prev_candles['Low'].min()
        
        if high_break and is_bullish:
            signal_type = "BULLISH_BREAKOUT"
            entry = candle['High']
            sl = candle['Low']
        elif low_break and not is_bullish:
            signal_type = "BEARISH_BREAKOUT"
            entry = candle['Low']
            sl = candle['High']
        else:
            signal_type = "SUPPORT_BOUNCE" if is_bullish else "RESISTANCE_BOUNCE"
            entry = candle['Close']
            sl = candle['Low'] if is_bullish else candle['High']
        
        # Calculate TP (2:1 RR)
        risk = abs(entry - sl)
        tp = entry + (2 * risk) if is_bullish else entry - (2 * risk)
        
        # Build signal dict
        signal = {
            "signal_type": signal_type,
            "instrument": "NIFTY",
            "entry_price": float(entry),
            "stop_loss": float(sl),
            "take_profit": float(tp),
            "price_level": float(entry),
            "confidence": np.random.uniform(60, 90),
            "volume_confirmed": candle['Volume'] > prev_candles['Volume'].mean() * 1.5,
            "momentum_confirmed": True,
            "risk_reward_ratio": 2.0,
            "timestamp": candle.name.isoformat(),
            "description": f"{signal_type} at {entry:.2f}"
        }
        
        # Build context (simplified)
        vwap = prev_candles['Close'].mean()  # Approximate VWAP
        ema20 = prev_candles['Close'].ewm(span=20).mean().iloc[-1]
        
        technical_context = {
            "higher_tf_context": {
                "trend_5m": "UP" if is_bullish else "DOWN",
                "trend_15m": "UP" if df.iloc[idx-12:idx]['Close'].mean() > df.iloc[idx-24:idx-12]['Close'].mean() else "DOWN",
                "trend_daily": "NEUTRAL",
                "vwap_5m": float(vwap),
                "ema20": float(ema20),
                "ema50": float(prev_candles['Close'].ewm(span=50).mean().iloc[-1] if len(prev_candles) >= 50 else ema20),
                "atr_percent": 0.08,
                "india_vix": 15.0
            }
        }
        
        # Option metrics (synthetic)
        option_metrics = {
            "pcr": np.random.uniform(0.8, 1.2),
            "iv": np.random.uniform(12, 18),
            "oi_change": {
                "sentiment": "BULLISH" if is_bullish else "BEARISH"
            }
        }
        
        # Calculate outcome (look ahead 10 candles)
        future_candles = df.iloc[idx+1:min(idx+11, len(df))]
        if len(future_candles) > 0:
            if is_bullish:
                tp_hit = (future_candles['High'] >= tp).any()
                sl_hit = (future_candles['Low'] <= sl).any()
            else:
                tp_hit = (future_candles['Low'] <= tp).any()
                sl_hit = (future_candles['High'] >= sl).any()
            
            # Determine outcome
            if tp_hit and not sl_hit:
                outcome = "WIN"
            elif sl_hit and not tp_hit:
                outcome = "LOSS"
            elif tp_hit and sl_hit:
                # Which came first?
                outcome = "WIN" if is_bullish == (future_candles['High'].idxmax() < future_candles['Low'].idxmin()) else "LOSS"
            else:
                outcome = "PENDING"
        else:
            outcome = "PENDING"
        
        signals.append({
            "signal": signal,
            "context": technical_context,
            "options": option_metrics,
            "outcome": outcome,
            "label": 1 if outcome == "WIN" else 0 if outcome == "LOSS" else None
        })
    
    logger.info(f"✅ Generated {len(signals)} signals")
    logger.info(f"   WIN: {sum(1 for s in signals if s['outcome'] == 'WIN')}")
    logger.info(f"   LOSS: {sum(1 for s in signals if s['outcome'] == 'LOSS')}")
    logger.info(f"   PENDING: {sum(1 for s in signals if s['outcome'] == 'PENDING')}")
    
    return signals


def test_feature_extraction(signals: list):
    """Test feature extractor with synthetic signals."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Feature Extraction")
    logger.info("="*70)
    
    try:
        # Extract features from first signal
        sample = signals[0]
        features = extract_features(
            sample["signal"],
            sample["context"],
            sample["options"]
        )
        
        logger.info(f"✅ Feature extraction successful")
        logger.info(f"   Features extracted: {len(features)}")
        logger.info(f"   Expected features: {len(get_feature_names())}")
        
        # Show sample features
        logger.info("\n   Sample features:")
        for key, value in list(features.items())[:10]:
            logger.info(f"     {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature extraction failed: {e}", exc_info=True)
        return False


def test_training_pipeline(signals: list):
    """Test training with synthetic data."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Training Pipeline")
    logger.info("="*70)
    
    try:
        # Filter to labeled signals only
        labeled_signals = [s for s in signals if s["label"] is not None]
        logger.info(f"Using {len(labeled_signals)} labeled samples")
        
        if len(labeled_signals) < 50:
            logger.warning("⚠️ Less than 50 samples, skipping training test")
            return False
        
        # Prepare training data
        from ml_module.feature_extractor import get_categorical_features
        
        features_list = []
        labels = []
        
        for sample in labeled_signals:
            features = extract_features(
                sample["signal"],
                sample["context"],
                sample["options"]
            )
            features_list.append(features)
            labels.append(sample["label"])
        
        X = pd.DataFrame(features_list)
        y = pd.Series(labels)
        
        # Convert categorical features to category dtype for LightGBM
        cat_features = get_categorical_features()
        for col in cat_features:
            if col in X.columns:
                X[col] = X[col].astype('category')
        
        logger.info(f"✅ Training data prepared")
        logger.info(f"   Shape: {X.shape}")
        logger.info(f"   Win rate: {y.mean():.2%}")
        
        # Quick model training
        import lightgbm as lgb
        
        params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 15,
            "learning_rate": 0.1,
            "verbose": -1
        }
        
        # Simple train/val split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=50,
            valid_sets=[val_data]
        )
        
        score = model.best_score["valid_0"]["auc"]
        logger.info(f"✅ Model trained successfully")
        logger.info(f"   Validation AUC: {score:.4f}")
        
        # Save model locally
        os.makedirs("models", exist_ok=True)
        model.save_model("models/test_model.txt")
        logger.info(f"✅ Model saved to models/test_model.txt")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return False


def test_predictor():
    """Test model loading and prediction."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Model Prediction")
    logger.info("="*70)
    
    try:
        # Note: This test uses local model file
        # For GCS test, you'd need valid bucket
        
        if not os.path.exists("models/test_model.txt"):
            logger.warning("⚠️ Test model not found, skipping predictor test")
            return False
        
        # Create predictor with mock GCS (will fail but we can test locally)
        logger.info("Testing local model loading...")
        
        import lightgbm as lgb
        model = lgb.Booster(model_file="models/test_model.txt")
        
        # Create sample features
        sample_features = {
            "signal_type": "BULLISH_BREAKOUT",
            "confidence": 75.0,
            "volume_confirmed": 1,
            "momentum_confirmed": 1,
            "risk_reward": 2.0,
            "stop_loss_pct": 1.0,
            "target_pct": 2.0,
            "trend_5m": "UP",
            "trend_15m": "UP",
            "trend_daily": "NEUTRAL",
            "trend_aligned_15m": 1,
            "trend_aligned_daily": 0,
            "distance_to_vwap_pct": 0.5,
            "distance_to_ema20_pct": 0.3,
            "distance_to_ema50_pct": 0.8,
            "above_vwap": 1,
            "above_ema20": 1,
            "atr_percent": 0.08,
            "pcr": 1.1,
            "iv": 15.0,
            "oi_sentiment": "BULLISH",
            "oi_aligned": 1,
            "hour_of_day": 10,
            "minute_of_hour": 30,
            "day_of_week": 2,
            "minutes_from_open": 75,
            "is_first_hour": 0,
            "is_last_hour": 0,
            "is_lunch_hour": 0,
            "india_vix": 15.0,
            "vix_regime": "NORMAL",
            "is_choppy": 0,
            "is_breakout": 1,
            "is_retest": 0,
            "is_reversal": 0,
            "is_inside_bar": 0
        }
        
        # Convert categorical features
        from ml_module.feature_extractor import get_categorical_features
        cat_features = get_categorical_features()
        
        df = pd.DataFrame([sample_features])
        for col in cat_features:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        prediction = model.predict(df)[0]
        
        logger.info(f"✅ Prediction successful")
        logger.info(f"   Input: BULLISH_BREAKOUT with high confidence")
        logger.info(f"   Predicted win probability: {prediction:.3f}")
        logger.info(f"   Decision: {'ACCEPT' if prediction >= 0.65 else 'REJECT'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    logger.info("="*70)
    logger.info("🧪 ML INFRASTRUCTURE TEST SUITE")
    logger.info("="*70)
    logger.info("Using last 7 days of 5m yfinance data (60-day limit)\n")
    
    # Step 1: Fetch data (7 days for 5m interval)
    df = fetch_historical_data(symbol="^NSEI", days=7)
    if df is None or df.empty:
        logger.error("❌ Failed to fetch historical data")
        return
    
    # Step 2: Generate synthetic signals
    signals = generate_synthetic_signals(df, num_signals=150)
    
    # Step 3: Run tests
    results = {}
    results["feature_extraction"] = test_feature_extraction(signals)
    results["training"] = test_training_pipeline(signals)
    results["prediction"] = test_predictor()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} - {test_name.replace('_', ' ').title()}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    logger.info(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        logger.info("\n🎉 All tests passed! ML infrastructure is ready.")
    else:
        logger.warning(f"\n⚠️ {total_tests - total_passed} test(s) failed")


if __name__ == "__main__":
    main()
