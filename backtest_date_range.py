#!/usr/bin/env python3
"""
Backtest NIFTY AI Trading Agent - Production-Identical Logic
Simulates real 5-minute execution with ALL production filters.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import logging
from datetime import datetime, timedelta
import pytz
import pandas as pd
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('./logs/backtest_range.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Import production components
from data_module.fetcher import get_data_fetcher
from analysis_module.technical import TechnicalAnalyzer
from config.settings import MIN_SIGNAL_CONFIDENCE
from main import NiftyTradingAgent

def run_backtest_for_date_range(start_date: str, end_date: str, instruments: List[str] = None):
    """
    Run backtest using PRODUCTION logic.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        instruments: List of instruments to test (default: ["NIFTY", "BANKNIFTY"])
    """
    if instruments is None:
        instruments = ["NIFTY", "BANKNIFTY"]
    
    logger.info("="*80)
    logger.info(f"🧪 BACKTEST (Production Logic): {start_date} to {end_date}")
    logger.info("="*80)
    logger.info(f"Instruments: {', '.join(instruments)}")
    MIN_SIGNAL_CONFIDENCE = 50  # Override for backtest visibility
    logger.info(f"Confidence Threshold: {MIN_SIGNAL_CONFIDENCE}% (Overridden for analysis)")
    logger.info("Filters: Choppy Session, Correlation, Time-of-Day, Duplicate Detection\n")
    
    # Initialize production agent (but don't send Telegram alerts)
    # We'll monkey-patch the telegram bot to not send messages
    agent = NiftyTradingAgent()
    
    # Disable Telegram notifications for backtest
    original_send = agent.telegram_bot.send_message
    original_send_breakout = agent.telegram_bot.send_breakout_alert
    original_send_retest = agent.telegram_bot.send_retest_alert
    original_send_inside_bar = agent.telegram_bot.send_inside_bar_alert
    
    def mock_send(*args, **kwargs):
        return True  # Pretend it sent successfully
    
    agent.telegram_bot.send_message = mock_send
    agent.telegram_bot.send_breakout_alert = mock_send
    agent.telegram_bot.send_retest_alert = mock_send
    agent.telegram_bot.send_retest_alert = mock_send
    agent.telegram_bot.send_inside_bar_alert = mock_send
    
    # Mock Option Chain Fetcher to prevent live API hits (invalid for backtest)
    def mock_fetch_oc(*args, **kwargs):
        return None  # Return None to simulate "no data" (neutral)
    
    agent.option_fetcher.fetch_option_chain = mock_fetch_oc
    
    # Results tracking
    all_signals = []
    event_counts = {
        "breakouts": 0,
        "breakdowns": 0,
        "retests": 0,
        "pin_bars": 0,
        "engulfing": 0,
        "inside_bars": 0
    }
    
    for instrument in instruments:
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 Testing {instrument}")
        logger.info(f"{'='*80}\n")
        
        try:
            # Fetch data
            logger.info(f"📡 Fetching data for {instrument}...")
            
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            days_needed = (end_dt - start_dt).days + 15
            
            df_5m = agent.fetcher.fetch_historical_data(
                instrument, 
                period=f"{days_needed}d", 
                interval="5m"
            )
            df_15m = agent.fetcher.fetch_historical_data(
                instrument,
                period=f"{days_needed + 5}d",
                interval="15m"
            )
            df_daily = agent.fetcher.fetch_historical_data(
                instrument,
                period="30d",
                interval="1d"
            )
            
            if df_5m is None or df_5m.empty:
                logger.error(f"❌ No 5m data for {instrument}")
                continue
            
            # Preprocess
            df_5m = agent.fetcher.preprocess_ohlcv(df_5m)
            df_15m = agent.fetcher.preprocess_ohlcv(df_15m) if df_15m is not None else None
            df_daily = agent.fetcher.preprocess_ohlcv(df_daily) if df_daily is not None else None
            
            logger.info(f"✅ Data loaded: {len(df_5m)} 5m candles\n")
            
            # Get candles in test range
            test_range_mask = (df_5m.index >= start_date) & (df_5m.index <= end_date + ' 23:59:59')
            test_indices = df_5m[test_range_mask].index
            
            if len(test_indices) == 0:
                logger.warning(f"⚠️  No data in specified date range for {instrument}")
                continue
            
            logger.info(f"📅 Simulating {len(test_indices)} time points\n")
            
            signals_found = []
            
            # Simulate production execution at each time point
            for i, current_time in enumerate(test_indices):
                # Get data available up to this point
                data_up_to_now = df_5m[df_5m.index <= current_time]
                
                if len(data_up_to_now) < 100:
                    continue
                
                # Mock NSE data at this time
                current_candle = data_up_to_now.iloc[-1]
                nse_data = {
                    "price": float(current_candle["close"]),
                    "lastPrice": float(current_candle["close"])
                }
                
                # Run PRODUCTION analysis
                analyzer = TechnicalAnalyzer(instrument)
                higher_tf_context = analyzer.get_higher_tf_context(df_15m, data_up_to_now, df_daily)
                analysis = analyzer.analyze_with_multi_tf(data_up_to_now, higher_tf_context)
                
                # Use PRODUCTION signal generation (includes ALL filters!)
                import main
                main.MIN_SIGNAL_CONFIDENCE = 50  # Patch the imported variable in main
                signals = agent._generate_signals(instrument, analysis, nse_data)
                
                # Track signals
                for sig in signals:
                    sig['timestamp'] = current_time
                    signals_found.append(sig)
                    
                    # Track events
                    sig_type = sig.get("signal_type", "")
                    if "BULLISH_BREAKOUT" in sig_type:
                        event_counts["breakouts"] += 1
                    elif "BEARISH_BREAKOUT" in sig_type or "BREAKDOWN" in sig_type:
                        event_counts["breakdowns"] += 1
                    elif any(x in sig_type for x in ["RETEST", "BOUNCE"]):
                        event_counts["retests"] += 1
                    elif "PIN_BAR" in sig_type:
                        event_counts["pin_bars"] += 1
                    elif "ENGULFING" in sig_type:
                        event_counts["engulfing"] += 1
                    elif "INSIDE_BAR" in sig_type:
                        event_counts["inside_bars"] += 1
                    
                    # Try to send through production pipeline (will be mocked)
                    # This tests duplicate detection and alert limits
                    agent._send_alert(sig)
                
                # Progress
                if (i + 1) % 50 == 0:
                    logger.info(f"   Analyzed {i+1}/{len(test_indices)} time points, {len(signals_found)} signals so far...")
            
            # Display results
            if signals_found:
                logger.info(f"\n🎯 {len(signals_found)} Signals (after all filters):\n")
                for i, sig in enumerate(signals_found[:20], 1):  # Show first 20
                    ts = sig.get('timestamp', 'N/A')
                    ts_str = ts.strftime('%Y-%m-%d %H:%M') if hasattr(ts, 'strftime') else str(ts)
                    logger.info(f"{i}. {sig.get('signal_type', 'N/A')} @ {ts_str}")
                    logger.info(f"   Entry: {sig.get('entry_price', 0):.2f} | Conf: {sig.get('confidence', 0):.0f}%\n")
                
                if len(signals_found) > 20:
                    logger.info(f"... and {len(signals_found) - 20} more\n")
                
                all_signals.extend(signals_found)
            else:
                logger.info(f"✅ No signals passed all filters for {instrument}\n")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {instrument}: {str(e)}", exc_info=True)
    
    # Restore original telegram methods
    agent.telegram_bot.send_message = original_send
    agent.telegram_bot.send_breakout_alert = original_send_breakout
    agent.telegram_bot.send_retest_alert = original_send_retest
    agent.telegram_bot.send_inside_bar_alert = original_send_inside_bar
    
    # Final Summary
    logger.info("\n" + "="*80)
    logger.info("📊 BACKTEST SUMMARY (Production Filters Applied)")
    logger.info("="*80)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Total Signals: {len(all_signals)}\n")
    
    logger.info("🎯 Event Counts:")
    logger.info(f"🚀 Breakouts: {event_counts['breakouts']}")
    logger.info(f"📉 Breakdowns: {event_counts['breakdowns']}")
    logger.info(f"🔄 Retests: {event_counts['retests']}")
    logger.info(f"🔨 Pin Bars: {event_counts['pin_bars']}")
    logger.info(f"🟢 Engulfing: {event_counts['engulfing']}")
    logger.info(f"📊 Inside Bars: {event_counts['inside_bars']}\n")
    
    if all_signals:
        logger.info("📋 Signals by Type:")
        by_type = {}
        for sig in all_signals:
            sig_type = sig.get('signal_type', 'UNKNOWN')
            if sig_type not in by_type:
                by_type[sig_type] = 0
            by_type[sig_type] += 1
        
        for sig_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   - {sig_type}: {count}")
        
        # Average metrics
        confidences = [s.get('confidence', 0) for s in all_signals if 'confidence' in s]
        rrs = [s.get('risk_reward_ratio', 0) for s in all_signals if 'risk_reward_ratio' in s]
        
        if confidences:
            logger.info(f"\n📈 Average Confidence: {sum(confidences)/len(confidences):.1f}%")
        if rrs:
            logger.info(f"📈 Average R:R: 1:{sum(rrs)/len(rrs):.1f}")
        
        logger.info(f"📅 Signals per Day: {len(all_signals) / 3:.1f}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ BACKTEST COMPLETE")
    logger.info("="*80 + "\n")
    
    logger.info("Alert Limits Status:")
    logger.info(f"   Recent alerts tracked: {len(agent.recent_alerts)}")
    stats = agent.persistence.get_daily_stats()
    logger.info(f"   Persistence stats: {stats.get('alerts_sent', 0)} alerts recorded\n")
    
    return all_signals, event_counts

if __name__ == "__main__":
    # Test for Nov 26-28, 2025
    signals, events = run_backtest_for_date_range(
        start_date="2025-11-26",
        end_date="2025-11-28",
        instruments=["NIFTY", "BANKNIFTY"]
    )
