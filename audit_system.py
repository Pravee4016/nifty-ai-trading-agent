
import sys
import os
import logging
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import NiftyTradingAgent
from config.settings import INSTRUMENTS

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SystemAudit")

def create_mock_ohlcv(rows=500, trend="UP"):
    """Generate mock OHLCV data."""
    data = []
    price = 25000.0
    for i in range(rows):
        if trend == "UP":
            price += 10 if i % 2 == 0 else -5
        else:
            price -= 10 if i % 2 == 0 else 5
            
        high = price + 20
        low = price - 20
        close = price + 5
        volume = 100000 + (i * 100)
        
        data.append({
            "timestamp": datetime.now(),
            "open": price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df

def audit_system():
    logger.info("🚀 Starting System Health Audit...")
    
    try:
        # 1. Initialize Agent
        logger.info("Step 1: Initializing Agent...")
        agent = NiftyTradingAgent()
        logger.info("✅ Agent Initialized")

        # 2. Mock Data Fetcher
        logger.info("Step 2: Mocking Data Sources...")
        agent.fetcher.fetch_nse_data = MagicMock(return_value={"price": 25500, "lastPrice": 25500})
        agent.fetcher.fetch_historical_data = MagicMock(side_effect=[
            create_mock_ohlcv(500, "UP"), # 5m
            create_mock_ohlcv(500, "UP"), # 15m
            create_mock_ohlcv(50, "UP")   # Daily
        ])
        agent.fetcher.preprocess_ohlcv = MagicMock(side_effect=lambda x: x) # Passthrough

        # 3. Mock Option Fetcher
        logger.info("Step 3: Mocking Option Chain...")
        agent.option_fetcher.fetch_option_chain = MagicMock(return_value={
            "records": {
                "expiryDates": ["12-Dec-2024"],
                "data": []
            },
            "filtered": {
                "data": []
            }
        })
        
        # 4. Mock AI & Telegram (Don't want real calls)
        agent.groq_analyzer.analyze_signal = MagicMock(return_value={"verdict": "BULLISH", "confidence": 85, "reasoning": "Audit Test"})
        agent.telegram_bot.send_message = MagicMock(return_value=True)
        agent.telegram_bot.send_alert = MagicMock(return_value=True)

        # 5. Run Analysis
        logger.info("Step 4: Running Full Analysis Cycle...")
        results = agent.run_analysis(instruments=["NIFTY"])
        
        logger.info(f"📊 Audit Results: {results}")
        
        if results["errors"] > 0:
            logger.error(f"❌ Audit Failed with {results['errors']} errors!")
            sys.exit(1)
            
        if results["signals_generated"] == 0:
             logger.warning("⚠️ No signals generated (Check filters?)")
        else:
             logger.info(f"✅ Generated {results['signals_generated']} signals")

        logger.info("✅ System Audit Passed!")

    except Exception as e:
        logger.error(f"❌ CRITICAL AUDIT FAILURE: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    audit_system()
