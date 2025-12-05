#!/usr/bin/env python3
"""
Backtest today's NIFTY50 data - bypasses main() market hours check.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

import logging
from datetime import datetime
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from data_module.fetcher import get_data_fetcher
from analysis_module.technical import TechnicalAnalyzer
from ai_module.groq_analyzer import get_analyzer
from telegram_module.bot_handler import get_bot

def run_backtest():
    """Run backtest on today's NIFTY data."""
    
    logger.info("="*80)
    logger.info("🧪 BACKTESTING TODAY'S NIFTY50 DATA (2025-12-01)")
    logger.info("="*80)
    
    # Initialize components
    fetcher = get_data_fetcher()
    ai_analyzer = get_analyzer()
    
    instrument = "NIFTY"
    
    try:
        # Step 1: Fetch NSE data (will likely fail)
        logger.info("\n📡 Step 1: Fetching NSE data...")
        nse_data = fetcher.fetch_nse_data(instrument)
        if not nse_data:
            logger.warning("⚠️  NSE data unavailable (expected) - continuing with yfinance")
            nse_data = {}
        
        # Step 2: Fetch historical data
        logger.info("\n📊 Step 2: Fetching historical data...")
        df_5m = fetcher.fetch_historical_data(instrument, period="5d", interval="5m")
        df_15m = fetcher.fetch_historical_data(instrument, period="10d", interval="15m")
        
        if df_5m is None or df_5m.empty:
            logger.error("❌ No 5m data available")
            return
        
        if df_15m is None or df_15m.empty:
            logger.error("❌ No 15m data available")
            return
        
        logger.info(f"✅ Got 5m data: {len(df_5m)} candles")
        logger.info(f"✅ Got 15m data: {len(df_15m)} candles")
        
        # Step 3: Preprocess
        logger.info("\n🔧 Step 3: Preprocessing data...")
        df_5m = fetcher.preprocess_ohlcv(df_5m)
        df_15m = fetcher.preprocess_ohlcv(df_15m)
        
        # Step 4: Run technical analysis
        logger.info("\n🔍 Step 4: Running technical analysis...")
        analyzer = TechnicalAnalyzer(instrument)
        higher_tf_context = analyzer.get_higher_tf_context(df_15m)
        analysis = analyzer.analyze_with_multi_tf(df_5m, higher_tf_context)
        
        logger.info(f"   Analysis complete:")
        logger.info(f"   - Breakout signal: {analysis.get('breakout_signal')}")
        logger.info(f"   - Breakdown signal: {analysis.get('breakdown_signal')}")
        logger.info(f"   - Support retest: {analysis.get('support_retest')}")
        logger.info(f"   - Resistance retest: {analysis.get('resistance_retest')}")
        
        # Step 5: Check for signals
        logger.info("\n🎯 Step 5: Checking for trading signals...")
        signals = []
        
        # Check breakout
        breakout = analysis.get("breakout_signal")
        if breakout and breakout.get("signal"):
            if breakout.get("confidence", 0) >= 70:
                signal_type = (
                    "BULLISH BREAKOUT" if breakout.get("direction") == "bullish"
                    else "BEARISH BREAKOUT"
                )
                signals.append({
                    "signal_type": signal_type,
                    "price_level": breakout.get("level"),
                    "confidence": breakout.get("confidence"),
                    "entry": breakout.get("entry_price"),
                    "sl": breakout.get("stop_loss"),
                    "tp": breakout.get("target"),
                })
                logger.info(f"   ✅ {signal_type} detected at {breakout.get('level'):.2f}")
        
        # Check breakdown (if separate logic exists)
        breakdown = analysis.get("breakdown_signal")
        if breakdown and breakdown.get("signal") and breakdown != breakout:
            if breakdown.get("confidence", 0) >= 70:
                signals.append({
                    "signal_type": "BEARISH BREAKDOWN",
                    "price_level": breakdown.get("level"),
                    "confidence": breakdown.get("confidence"),
                    "entry": breakdown.get("entry_price"),
                    "sl": breakdown.get("stop_loss"),
                    "tp": breakdown.get("target"),
                })
                logger.info(f"   ✅ BEARISH BREAKDOWN detected at {breakdown.get('level'):.2f}")
        
        # Check retests
        retest_signal = analysis.get("retest_signal")
        if retest_signal:
            signals.append({
                "signal_type": retest_signal.signal_type.value,
                "price_level": retest_signal.price_level,
                "confidence": retest_signal.confidence,
                "entry": retest_signal.entry_price,
                "sl": retest_signal.stop_loss,
                "tp": retest_signal.take_profit,
            })
            logger.info(f"   ✅ {retest_signal.signal_type.value} detected")
        
        # Display results
        logger.info("\n" + "="*80)
        logger.info("📊 BACKTEST RESULTS")
        logger.info("="*80)
        logger.info(f"Date: 2025-12-01")
        logger.info(f"Instrument: {instrument}")
        logger.info(f"Total Signals: {len(signals)}")
        
        if signals:
            logger.info(f"\n🎯 SIGNALS THAT SHOULD HAVE TRIGGERED ALERTS:\n")
            for i, sig in enumerate(signals, 1):
                logger.info(f"{i}. {sig['signal_type']}")
                logger.info(f"   Price Level: {sig.get('price_level', 'N/A')}")
                logger.info(f"   Confidence: {sig.get('confidence', 'N/A')}%")
                if 'entry' in sig:
                    logger.info(f"   Entry: {sig.get('entry'):.2f}")
                    logger.info(f"   SL: {sig.get('sl'):.2f}")
                    logger.info(f"   TP: {sig.get('tp'):.2f}")
                logger.info("")
        else:
            logger.info("\n✅ NO SIGNALS DETECTED")
            logger.info("This means:")
            logger.info("  - No clear breakout/breakdown patterns occurred today")
            logger.info("  - OR patterns didn't meet the 70% confidence threshold")
            logger.info("  - System is working correctly - no false alerts!")
        
        logger.info("\n" + "="*80)
        
        # Print data quality info
        logger.info("\n📈 DATA QUALITY:")
        logger.info(f"Latest 5m candle time: {df_5m.index[-1]}")
        logger.info(f"Latest 5m close: {df_5m.iloc[-1]['close']:.2f}")
        logger.info(f"Today's range: {df_5m.iloc[-1]['high']:.2f} - {df_5m.iloc[-1]['low']:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Backtest failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    run_backtest()
