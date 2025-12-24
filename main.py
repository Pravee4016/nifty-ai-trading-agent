"""
Nifty AI Trading Agent - Main Entry Point
Uses AppFactory pattern and centralized logging.
"""
import logging
import pytz
from datetime import datetime, time
import argparse

# Config
from config.settings import TIME_ZONE, DEBUG_MODE
from config.logging_config import setup_logging

# App Bootstrap
from app.bootstrap import create_agent

# Dependencies for standalone tasks
from data_module.persistence import get_persistence
from telegram_module.bot_handler import get_bot

# Setup Logger
logger = setup_logging(__name__)

# ------------------------------------------------------
# Helper Functions
# ------------------------------------------------------

def _is_market_hours_quick():
    """Quick market hours check for early exit (Cloud Function optimization)."""
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).time()
    today = datetime.now(ist).weekday()
    # Allow execution until 16:30 for EOD activities (Summary, Market Closed Alert)
    # Weekdays only (0-4)
    if today > 4: return False
    return time(9, 15) <= now <= time(16, 30)

def check_and_send_market_closed_alert():
    """
    Check if 'Market Closed' alert has been sent today. 
    If not, and it is after 15:30 IST, send it once.
    """
    try:
        tz = pytz.timezone(TIME_ZONE)
        now = datetime.now(tz)
        
        # Only check if it's after market close (15:30)
        # We use 15:30 inclusive in case the job runs exactly then
        if now.time() >= time(15, 30):
            persistence = get_persistence()
            stats = persistence.get_daily_stats()
            
            if not stats.get("market_closed_msg_sent"):
                logger.info("🌙 Market Closed - Sending one-time alert")
                bot = get_bot()
                bot.send_message("🌙 <b>Market Closed</b> - Analysis Paused")
                persistence.increment_stat("market_closed_msg_sent")
            else:
                logger.debug("🌙 Market Closed - Alert already sent today")
                
    except Exception as e:
        logger.error(f"❌ Failed to check/send market closed alert: {e}")

# ------------------------------------------------------
# Main Execution
# ------------------------------------------------------

def main():
    """Main entry point for local run or cloud function."""
    logger.info("🚀 Agent starting...")
    
    # Use Factory to create agent
    try:
        agent = create_agent()
    except Exception as e:
        logger.critical("🔥 Agent initialization failed. Exiting.")
        return

    logger.info("\n🔌 Testing external connections...")
    if not agent.ai_analyzer.test_connection():
        logger.error("❌ Groq API connection failed")
        agent.telegram_bot.send_error_notification(
            "Groq API connection failed"
        )
        return

    if not agent.telegram_bot.test_connection():
        logger.error("❌ Telegram connection failed")
        return

    logger.info("✅ All connections successful\n")

    # Check for scheduled messages (Startup, Market Context, EOD Summary)
    # Must run outside is_market_hours() to catch EOD summary at 15:31+
    agent.check_scheduled_messages()

    if agent.is_market_hours():
        # Run Analysis
        results = agent.run_analysis()
    else:
        logger.info("⏰ Outside market hours - skipping analysis")
        check_and_send_market_closed_alert()

    stats = agent.get_statistics()
    logger.info("\n📈 STATISTICS")
    logger.info(f"Alerts Sent: {stats['alerts_sent']}")
    logger.info(f"ML Predictions: {stats.get('ml_predictions', 0)}")
    
    # Handle AI usage stats (supports both old Groq format and new factory format)
    ai_usage = stats.get('ai_usage', {})
    if ai_usage:
        # Check if it's the new hybrid/factory format
        if 'mode' in ai_usage:
            logger.info(f" AI Mode: {ai_usage['mode']}")
        # Try to get tokens used (may not exist for all providers)
        tokens = ai_usage.get('tokens_used', 'N/A')
        if tokens != 'N/A':
            logger.info(f" AI Usage: {tokens} tokens used")
    
    logger.info(f"Session Time: {stats.get('execution_time', 0):.2f}s")


def cloud_function_handler(request):
    """Entry point for Google Cloud Functions."""
    # Early exit for outside market hours (avoid full initialization)
    if not _is_market_hours_quick():
        # Check if we need to send the "Market Closed" alert before exiting
        check_and_send_market_closed_alert()
        return {"status": "skipped", "message": "Outside market hours"}
    
    logger.info("☁️  Cloud Function triggered")
    main()
    return {"status": "success", "message": "Analysis completed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nifty AI Trading Agent")
    parser.add_argument(
        "--once", action="store_true", help="Run analysis once"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test connections only"
    )

    args = parser.parse_args()

    if args.test:
        logger.info("🔧 Running Connection Tests...")
        try:
            agent = create_agent()
            logger.info("Testing connections...")
            if agent.ai_analyzer.test_connection():
                logger.info("✅ AI Connection OK")
            else:
                logger.error("❌ AI Connection FAILED")
                
            if agent.telegram_bot.test_connection():
                 logger.info("✅ Telegram Connection OK")
            else:
                 logger.error("❌ Telegram Connection FAILED")
                 
        except Exception as e:
            logger.error(f"❌ Test initialization failed: {e}")
    else:
        main()
