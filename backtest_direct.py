#!/usr/bin/env python3
"""
Direct backtest of today's NIFTY50 data - bypasses market hours check.
"""

import os
os.environ['DRY_RUN'] = 'False'  # Enable real analysis

import logging
from datetime import datetime
import pytz

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Import components directly
from main import NiftyTradingAgent

logger.info("="*70)
logger.info("🧪 DIRECT BACKTEST - TODAY'S NIFTY50 DATA (2025-12-01)")
logger.info("="*70)

# Create agent
agent = NiftyTradingAgent()

# Run analysis on NIFTY
logger.info("\n📊 Running analysis on NIFTY for today...")
result = agent.analyze_instrument("NIFTY")

# Display results
logger.info("\n" + "="*70)
logger.info("📊 BACKTEST RESULTS")
logger.info("="*70)

logger.info(f"Signals Generated: {result.get('signals_count', 0)}")
logger.info(f"Alerts Sent: {result.get('alerts_sent', 0)}")
logger.info(f"Errors: {len(result.get('errors', []))}")

signals = result.get('signals', [])
if signals:
    logger.info(f"\n🎯 {len(signals)} SIGNAL(S) DETECTED:\n")
    for i, sig in enumerate(signals, 1):
        logger.info(f"{i}. {sig.get('signal_type')}")
        logger.info(f"   Level: {sig.get('price_level'):.2f}")
        logger.info(f"   Entry: {sig.get('entry'):.2f} | SL: {sig.get('sl'):.2f} | TP: {sig.get('tp'):.2f}")
        logger.info(f"   Confidence: {sig.get('confidence')}%")
        
        ai = sig.get('ai_analysis', {})
        if ai:
            logger.info(f"   AI: {ai.get('recommendation')} ({ai.get('confidence')}% confidence)")
            logger.info(f"   Summary: {ai.get('summary', '')[:80]}...")
        logger.info("")
else:
    logger.info("\n⚠️  NO SIGNALS DETECTED")

if result.get('errors'):
    logger.info(f"\n❌ ERRORS: {result['errors']}")

logger.info("\n" + "="*70)
agent.ai_analyzer.print_usage_stats()
