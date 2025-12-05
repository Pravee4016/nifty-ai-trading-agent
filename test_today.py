#!/usr/bin/env python3
"""
Backtest script for today's NIFTY50 data to verify alert system.
"""

import logging
import sys
from datetime import datetime
import pytz

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

from main import NiftyTradingAgent

def main():
    """Run analysis on today's NIFTY50 data."""
    
    logger.info("="*70)
    logger.info("🧪 BACKTESTING TODAY'S NIFTY50 DATA")
    logger.info("="*70)
    
    # Initialize agent
    agent = NiftyTradingAgent()
    
    # Analyze NIFTY
    instrument = "NIFTY"
    logger.info(f"\n📊 Analyzing {instrument} for 2025-12-01...\n")
    
    result = agent.analyze_instrument(instrument)
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("📊 BACKTEST RESULTS")
    logger.info("="*70)
    logger.info(f"Instrument: {instrument}")
    logger.info(f"Date: {datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d')}")
    logger.info(f"\nSignals Generated: {result.get('signals_count', 0)}")
    logger.info(f"Alerts Sent: {result.get('alerts_sent', 0)}")
    logger.info(f"Errors: {len(result.get('errors', []))}")
    
    # Display signals details
    signals = result.get('signals', [])
    if signals:
        logger.info(f"\n🎯 SIGNALS DETECTED ({len(signals)}):")
        logger.info("-"*70)
        for i, sig in enumerate(signals, 1):
            logger.info(f"\n{i}. {sig.get('signal_type', 'UNKNOWN')}")
            logger.info(f"   Price Level: {sig.get('price_level', 'N/A')}")
            logger.info(f"   Entry: {sig.get('entry', 'N/A')}")
            logger.info(f"   SL: {sig.get('sl', 'N/A')}")
            logger.info(f"   TP: {sig.get('tp', 'N/A')}")
            logger.info(f"   Confidence: {sig.get('confidence', 'N/A')}")
            
            # AI Analysis
            ai = sig.get('ai_analysis', {})
            if ai:
                logger.info(f"   AI Recommendation: {ai.get('recommendation', 'N/A')}")
                logger.info(f"   AI Confidence: {ai.get('confidence', 'N/A')}%")
                logger.info(f"   AI Summary: {ai.get('summary', 'N/A')[:100]}...")
    else:
        logger.info("\n⚠️  NO SIGNALS DETECTED")
        logger.info("This could mean:")
        logger.info("  1. No breakout/breakdown patterns occurred today")
        logger.info("  2. Patterns didn't meet confidence threshold")
        logger.info("  3. System is working correctly but market was neutral")
    
    # Display errors
    if result.get('errors'):
        logger.info(f"\n❌ ERRORS:")
        for err in result['errors']:
            logger.info(f"   - {err}")
    
    logger.info("\n" + "="*70)
    
    # Print Groq usage stats
    agent.ai_analyzer.print_usage_stats()
    
    return result

if __name__ == "__main__":
    try:
        result = main()
        
        # Exit with appropriate code
        if result.get('errors'):
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"❌ Backtest failed: {str(e)}", exc_info=True)
        sys.exit(1)
