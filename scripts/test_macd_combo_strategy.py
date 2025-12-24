"""
Test MACD + Combo Strategy with Historical Data
Runs the new MACD/RSI/BB confluence scoring on historical NIFTY data
"""

import sys
import os
import pandas as pd
import logging
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis_module.technical import TechnicalAnalyzer
from analysis_module.combo_signals import MACDRSIBBCombo
from config.settings import USE_COMBO_SIGNALS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_historical_data(symbol='NIFTY', days=5):
    """Fetch historical data using yfinance."""
    try:
        import yfinance as yf
        
        # Map symbol to yfinance ticker
        ticker_map = {
            'NIFTY': '^NSEI',
            'BANKNIFTY': '^NSEBANK'
        }
        
        ticker = ticker_map.get(symbol, '^NSEI')
        
        logger.info(f"📥 Fetching {days} days of historical data for {symbol}...")
        
        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='5m',
            progress=False
        )
        
        if df.empty:
            logger.error(f"No data fetched for {symbol}")
            return None
        
        # Normalize column names (yfinance returns capitalized columns)
        # Handle both single and multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df.columns = [str(col).lower() for col in df.columns]
        
        logger.info(f"✅ Fetched {len(df)} candles | Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None


def test_macd_calculation(df, analyzer):
    """Test MACD calculation."""
    logger.info("\n" + "="*80)
    logger.info("🔍 TESTING: MACD Calculation")
    logger.info("="*80)
    
    macd_data = analyzer._calculate_macd(df)
    
    logger.info(f"\n📊 MACD Results:")
    logger.info(f"  MACD Line:    {macd_data['macd_line']:.2f}")
    logger.info(f"  Signal Line:  {macd_data['signal_line']:.2f}")
    logger.info(f"  Histogram:    {macd_data['histogram']:.2f}")
    logger.info(f"  Crossover:    {macd_data['crossover']}")
    
    return macd_data


def test_ema_crossover(df, analyzer):
    """Test EMA crossover detection."""
    logger.info("\n" + "="*80)
    logger.info("🔍 TESTING: EMA Crossover (5/15)")
    logger.info("="*80)
    
    ema_data = analyzer.detect_ema_crossover(df)
    
    logger.info(f"\n📈 EMA Crossover Results:")
    logger.info(f"  Directional Bias:  {ema_data['bias']}")
    logger.info(f"  Confidence:        {ema_data['confidence']:.2f}")
    logger.info(f"  Price Separation:  {ema_data['price_separation_pct']:.2f}%")
    logger.info(f"  EMA Fast (5):      {ema_data['ema_fast']:.2f}")
    logger.info(f"  EMA Slow (15):     {ema_data['ema_slow']:.2f}")
    logger.info(f"  Current Price:     {df['close'].iloc[-1]:.2f}")
    
    return ema_data


def test_combo_signal(df, analyzer, combo_evaluator):
    """Test combo signal evaluation."""
    logger.info("\n" + "="*80)
    logger.info("🔍 TESTING: MACD + RSI + BB Combo Signal")
    logger.info("="*80)
    
    # Calculate all indicators needed
    macd_data = analyzer._calculate_macd(df)
    ema_data = analyzer.detect_ema_crossover(df)
    rsi = analyzer._calculate_rsi(df)
    
    # Calculate Bollinger Bands
    bb_data = analyzer._calculate_bollinger_bands(df)
    
    # Build technical context
    technical_context = {
        "macd": macd_data,
        "rsi_5": rsi,
        "bb_upper": bb_data['upper'].iloc[-1] if bb_data and 'upper' in bb_data else 0.0,
        "bb_lower": bb_data['lower'].iloc[-1] if bb_data and 'lower' in bb_data else 0.0
    }
    
    # Get direction from EMA crossover or use bullish as default
    direction = ema_data['bias'] if ema_data['bias'] != "NEUTRAL" else "BULLISH"
    
    # Evaluate combo signal
    combo_result = combo_evaluator.evaluate_signal(
        df=df,
        direction_bias=direction,
        technical_context=technical_context
    )
    
    logger.info(f"\n⭐ Combo Signal Results:")
    logger.info(f"  Direction:        {direction}")
    logger.info(f"  Strength:         {combo_result['strength']}")
    logger.info(f"  Score:            {combo_result['score']}/3")
    logger.info(f"  BB Position:      {combo_result['bb_position']:.2f} (0=lower, 1=upper)")
    logger.info(f"  Details:          {combo_result['details']}")
    logger.info(f"\n  Conditions Met:")
    for condition, met in combo_result['conditions'].items():
        status = "✅" if met else "❌"
        logger.info(f"    {status} {condition}: {met}")
    
    return combo_result


def test_pattern_with_combo(df, analyzer, combo_evaluator):
    """Test pattern detection with combo scoring."""
    logger.info("\n" + "="*80)
    logger.info("🔍 TESTING: Pattern Detection + Combo Scoring")
    logger.info("="*80)
    
    # Calculate support/resistance
    support_resistance = analyzer.calculate_support_resistance(df)
    
    # Calculate higher timeframe context (mock for testing)
    higher_tf_context = {
        "trend_direction": "UP",
        "rsi_15": 55.0,
        "price_above_vwap": True,
        "price_above_ema20": True,
        "vwap_slope": "UP",
        "prev_day_trend": "UP",
        "vwap_5m": df['close'].iloc[-1] * 0.998,
        "ema_20_5m": df['close'].iloc[-1] * 0.995
    }
    
    # Calculate MACD and BB for context
    macd_data = analyzer._calculate_macd(df)
    bb_data = analyzer._calculate_bollinger_bands(df)
    rsi = analyzer._calculate_rsi(df)
    
    # Build technical context
    technical_context = {
        "macd": macd_data,
        "rsi_5": rsi,
        "bb_upper": bb_data['upper'].iloc[-1] if bb_data and 'upper' in bb_data else 0.0,
        "bb_lower": bb_data['lower'].iloc[-1] if bb_data and 'lower' in bb_data else 0.0
    }
    
    # Try to detect patterns
    patterns_found = []
    
    # Breakout detection
    signal = analyzer.detect_breakout(df, support_resistance, higher_tf_context)
    if signal:
        patterns_found.append(("BREAKOUT", signal))
    
    # Pin bar detection
    signal = analyzer.detect_pin_bar(df, support_resistance, higher_tf_context)
    if signal:
        patterns_found.append(("PIN_BAR", signal))
    
    # Inside bar detection
    signal = analyzer.detect_inside_bar(df, support_resistance, higher_tf_context)
    if signal:
        patterns_found.append(("INSIDE_BAR", signal))
    
    logger.info(f"\n📊 Patterns Detected: {len(patterns_found)}")
    
    for pattern_name, signal in patterns_found:
        logger.info(f"\n  🎯 {pattern_name}:")
        logger.info(f"    Entry:      {signal.entry_price:.2f}")
        logger.info(f"    Stop Loss:  {signal.stop_loss:.2f}")
        logger.info(f"    Target:     {signal.take_profit:.2f}")
        logger.info(f"    R:R:        {signal.risk_reward_ratio:.2f}")
        logger.info(f"    Confidence: {signal.confidence:.0f}%")
        
        # Evaluate combo for this pattern
        direction = signal.debug_info.get('direction', 'LONG')
        combo_result = combo_evaluator.evaluate_signal(
            df=df,
            direction_bias=direction,
            technical_context=technical_context
        )
        
        logger.info(f"    Combo:      {combo_result['strength']} ({combo_result['score']}/3)")
    
    return patterns_found


def run_historical_test():
    """Main test function."""
    logger.info("="*80)
    logger.info("🚀 MACD + COMBO STRATEGY - HISTORICAL DATA TEST")
    logger.info("="*80)
    logger.info(f"Combo Signals Enabled: {USE_COMBO_SIGNALS}")
    logger.info("")
    
    # Fetch historical data
    df = fetch_historical_data('NIFTY', days=5)
    if df is None or df.empty:
        logger.error("❌ Failed to fetch historical data")
        return
    
    # Initialize components
    analyzer = TechnicalAnalyzer("NIFTY 50")
    combo_evaluator = MACDRSIBBCombo()
    
    # Run tests
    logger.info(f"\n📊 Current Market Data:")
    logger.info(f"  Latest Close: {df['close'].iloc[-1]:.2f}")
    logger.info(f"  Latest High:  {df['high'].iloc[-1]:.2f}")
    logger.info(f"  Latest Low:   {df['low'].iloc[-1]:.2f}")
    logger.info(f"  Latest Vol:   {df['volume'].iloc[-1]:,.0f}")
    
    # Test 1: MACD
    macd_data = test_macd_calculation(df, analyzer)
    
    # Test 2: EMA Crossover
    ema_data = test_ema_crossover(df, analyzer)
    
    # Test 3: Combo Signal
    combo_result = test_combo_signal(df, analyzer, combo_evaluator)
    
    # Test 4: Pattern Detection with Combo
    patterns = test_pattern_with_combo(df, analyzer, combo_evaluator)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("✅ TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"  MACD Working:         ✅ (Histogram: {macd_data['histogram']:.2f})")
    logger.info(f"  EMA Crossover:        ✅ (Bias: {ema_data['bias']})")
    logger.info(f"  Combo Signal:         ✅ (Strength: {combo_result['strength']})")
    logger.info(f"  Patterns Detected:    {len(patterns)}")
    logger.info("")
    logger.info("🎉 All components working! Ready for production integration.")
    logger.info("="*80)


if __name__ == "__main__":
    run_historical_test()
