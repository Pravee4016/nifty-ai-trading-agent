"""
Test script to send a test alert to Telegram
"""

import sys
import os
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from telegram_module.bot_handler import get_bot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def send_test_alert():
    """Send a test signal alert to Telegram"""
    logger.info("🧪 Starting Telegram Test...")
    
    # Get bot instance
    bot = get_bot()
    
    # Test 1: Connection test
    logger.info("\n📡 Test 1: Connection Test")
    connection_ok = bot.test_connection()
    
    if not connection_ok:
        logger.error("❌ Connection test failed. Check your bot token and chat ID.")
        return False
    
    # Test 2: Send a simple message
    logger.info("\n📨 Test 2: Simple Message")
    message_ok = bot.send_message(
        "🧪 <b>Test Alert</b>\n\n"
        "This is a test message from the Nifty AI Trading Agent.\n\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    if not message_ok:
        logger.error("❌ Message send failed.")
        return False
    
    # Test 3: Send a sample trading signal
    logger.info("\n📊 Test 3: Sample Trading Signal")
    
    sample_signal = {
        "instrument": "NIFTY 50",
        "signal_type": "BULLISH RETEST",
        "entry_price": 25000.00,
        "stop_loss": 24950.00,
        "take_profit": 25075.00,
        "take_profit_2": 25125.00,
        "take_profit_3": 25150.00,
        "risk_reward_ratio": 1.5,
        "confidence": 72.0,
        "score": 68,
        "price_level": 24980.00,
        "description": "Strong bounce at PDL support with high volume confirmation",
        "score_reasons": [
            "PDL Support",
            "High Volume (1.8x)",
            "15m Uptrend",
            "RSI Bullish (62)"
        ],
        "ai_analysis": {
            "verdict": "STRONG BUY",
            "reasoning": "Price showing strong bounce at key support with volume confirmation. Risk:reward favorable.",
            "confidence": 75
        }
    }
    
    signal_ok = bot.send_retest_alert(sample_signal)
    
    if not signal_ok:
        logger.error("❌ Signal alert send failed.")
        return False
    
    # Test 4: System health message
    logger.info("\n🏥 Test 4: System Health Message")
    health_message = (
        "✅ <b>SYSTEM HEALTH CHECK</b>\n\n"
        "<b>📊 Cloud Run Services:</b>\n"
        "• nifty-scalping-agent: ✅ Running (asia-south1)\n"
        "• eurusd-london-agent: ✅ Running (us-central1)\n\n"
        "<b>⏰ Cloud Scheduler:</b>\n"
        "• nifty-data-fetch: ✅ Enabled\n"
        "• Schedule: */5 9-15 * * 1-5 (IST)\n\n"
        "<b>🧪 Unit Tests:</b>\n"
        "• Retest validation: 3/3 ✅\n"
        "• Target capping: 2/2 ✅\n"
        "• RVOL calculation: 2/2 ✅\n"
        "• Total: 7/7 passed ✅\n\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}"
    )
    
    health_ok = bot.send_message(health_message)
    
    if not health_ok:
        logger.error("❌ Health message send failed.")
        return False
    
    logger.info("\n" + "="*60)
    logger.info("✅ ALL TELEGRAM TESTS PASSED!")
    logger.info("="*60)
    logger.info("\n📊 Summary:")
    logger.info("   ✅ Connection test passed")
    logger.info("   ✅ Simple message sent")
    logger.info("   ✅ Trading signal alert sent")
    logger.info("   ✅ System health message sent")
    logger.info("\n🎉 Telegram integration is working correctly!")
    
    return True


if __name__ == "__main__":
    try:
        success = send_test_alert()
        if success:
            print("\n✅ Test completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Test failed!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
