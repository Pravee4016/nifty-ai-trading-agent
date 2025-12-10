
import os
import sys
import logging
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from telegram_module.bot_handler import TelegramBot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_broadcast():
    print("--- Testing Telegram Broadcast ---")
    
    bot = TelegramBot()
    
    # 1. Simple Message
    print("\n1. Sending Simple Message...")
    success = bot.send_message("🔔 <b>Test Broadcast</b>\n\nThis message should appear in:\n1. Your Private Chat\n2. Your Channel (@NiftyAlertsPr)")
    
    if success:
        print("✅ Simple Message Sent")
    else:
        print("❌ Simple Message Failed")
        
    # 2. Mock Alert
    print("\n2. Sending Mock Trading Alert...")
    mock_signal = {
        "instrument": "NIFTY 50",
        "signal_type": "BREAKOUT_TEST",
        "entry_price": 25100.00,
        "stop_loss": 25050.00,
        "take_profit": 25200.00,
        "risk_reward_ratio": 2.0,
        "confidence": 85.0,
        "score": 90,
        "score_reasons": ["High Volume", "Option Chain Bullish"],
        "description": "Test breakout pattern for verification."
    }
    
    success_alert = bot.send_breakout_alert(mock_signal)
    
    if success_alert:
        print("✅ Mock Alert Sent")
    else:
        print("❌ Mock Alert Failed")

if __name__ == "__main__":
    test_broadcast()
