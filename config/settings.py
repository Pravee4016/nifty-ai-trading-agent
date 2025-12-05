"""
Configuration & Settings Module
All API keys, thresholds, and constants in one place for easy debugging
"""

import os
from dotenv import load_dotenv
from enum import Enum

# Load environment variables
load_dotenv()

# ============================================================================
# API KEYS & CREDENTIALS
# ============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_KEY_HERE")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")

# ============================================================================
# MARKET PARAMETERS
# ============================================================================


class Instrument(Enum):
    """Trading instruments"""

    NIFTY50 = "NIFTY 50"
    BANKNIFTY = "BANKNIFTY"
    FINNIFTY = "FINNIFTY"


INSTRUMENTS = {
    "NIFTY": {
        "symbol": "NIFTY 50",
        "display_name": "Nifty 50",
        "tick_size": 0.05,
        "lot_size": 75,
        "active": True,
    },
    "BANKNIFTY": {
        "symbol": "BANKNIFTY",
        "display_name": "Bank Nifty",
        "tick_size": 0.05,
        "lot_size": 15,
        "active": False,
    },
    "FINNIFTY": {
        "symbol": "FINNIFTY",
        "display_name": "Fin Nifty",
        "tick_size": 0.05,
        "lot_size": 40,
        "active": False,  # Enable when ready
    },
}

# Trading hours (IST)
MARKET_OPEN_TIME = "09:15"  # Market opens
ANALYSIS_START_TIME = os.getenv("ANALYSIS_START_TIME", "09:05")  # Analysis trigger
MARKET_CLOSE_TIME = "15:30"  # Market closes

# Timeframes for analysis
TIMEFRAMES = {
    "5MIN": "5m",
    "15MIN": "15m",
    "1HOUR": "1h",
    "1DAY": "d",
}

PRIMARY_TIMEFRAME = "5MIN"  # Main breakout detection
CONFIRMATION_TIMEFRAME = "15MIN"  # Trend confirmation

# ============================================================================
# TECHNICAL ANALYSIS THRESHOLDS
# ============================================================================

# Volume Configuration
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", 1.5))  # 1.5x avg volume
VOLUME_PERIOD = 20  # candles lookback

# Breakout Configuration
BREAKOUT_CONFIRMATION_CANDLES = 2
FALSE_BREAKOUT_RETRACEMENT = float(os.getenv("MAX_FALSE_BREAKOUT_PERCENT", 0.5))
RETEST_ZONE_PERCENT = float(os.getenv("RETEST_ZONE_PERCENT", 0.3))

# Support/Resistance Configuration
SR_CLUSTER_TOLERANCE = 0.1  # 0.1% difference = same cluster
MIN_SR_TOUCHES = 2
LOOKBACK_BARS = 100

# Momentum Configuration
MIN_RSI_BULLISH = int(os.getenv("MIN_MOMENTUM_RSI", 60))
MAX_RSI_BEARISH = int(os.getenv("MAX_MOMENTUM_RSI", 40))
RSI_PERIOD = 14

# ATR (for dynamic SL/TP)
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.5

# Moving Averages
EMA_SHORT = 9
EMA_LONG = 21
SMA_VOLUME = 20

# MACD Configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands
BB_PERIOD = 20
BB_STD_DEV = 2

# Stochastic Configuration
STOCH_PERIOD = 14
STOCH_SMOOTH_K = 3
STOCH_SMOOTH_D = 3

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

MIN_RISK_REWARD_RATIO = 1.5
MAX_DAILY_TRADES = 5
MIN_SIGNAL_CONFIDENCE = int(os.getenv("MIN_SIGNAL_CONFIDENCE", 65))

DEFAULT_POSITION_SIZE = 1
MAX_POSITION_SIZE = 3

# ============================================================================
# AI & GROQ CONFIGURATION
# ============================================================================

GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", 400))
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", 0.3))

# Approximate token budget (for your own tracking; not enforced by API)
GROQ_REQUESTS_PER_DAY = 30000
GROQ_REQUESTS_PER_MINUTE = 300

# ============================================================================
# TELEGRAM CONFIGURATION
# ============================================================================

TELEGRAM_MAX_MESSAGE_LENGTH = 4000
INCLUDE_CHARTS_IN_ALERT = (
    os.getenv("INCLUDE_CHARTS_IN_ALERT", "true").lower() == "true"
)
INCLUDE_AI_SUMMARY_IN_ALERT = (
    os.getenv("INCLUDE_AI_SUMMARY_IN_ALERT", "true").lower() == "true"
)

ALERT_TYPES = {
    "BREAKOUT": "🚀 BREAKOUT",
    "BREAKOUT_CONFIRMED": "✅ BREAKOUT CONFIRMED",
    "FALSE_BREAKOUT": "⚠️ FALSE BREAKOUT",
    "RETEST": "🎯 RETEST SETUP",
    "INSIDE_BAR": "📊 INSIDE BAR SETUP",
    "BREAKDOWN": "📉 BREAKDOWN",
    "SUPPORT_HIT": "🛡️ SUPPORT HIT",
    "RESISTANCE_HIT": "🔴 RESISTANCE HIT",
}

# ============================================================================
# SCHEDULING CONFIGURATION
# ============================================================================

TIME_ZONE = os.getenv("TIME_ZONE", "Asia/Kolkata")
SCHEDULE_CRON = "5 9 * * MON-FRI"
INTRADAY_MONITORING_INTERVAL = 5  # minutes

# ============================================================================
# DATA STORAGE & CACHING
# ============================================================================

CACHE_DIR = "./cache"
CACHE_DATA_TTL = 3600  # seconds

LOG_DIR = "./logs"
LOG_LEVEL = os.getenv("LOGLEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# CLOUD DEPLOYMENT
# ============================================================================

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
CLOUD_FUNCTION_REGION = os.getenv("CLOUD_FUNCTION_REGION", "us-central1")
FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE", "")

DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "LOCAL")  # LOCAL, GCP, RENDER
ENABLE_CLOUD_LOGGING = DEPLOYMENT_MODE != "LOCAL"

# ============================================================================
# DEBUG & DEVELOPMENT
# ============================================================================

DEBUG_MODE = os.getenv("DEBUG_MODE", "False") == "True"

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Daily Alert Limits
MAX_ALERTS_PER_DAY = int(os.getenv("MAX_ALERTS_PER_DAY", "999"))  # Effectively unlimited (rely on other filters)
MAX_ALERTS_PER_TYPE = int(os.getenv("MAX_ALERTS_PER_TYPE", "10"))  # Max per signal type
MAX_ALERTS_PER_INSTRUMENT = int(os.getenv("MAX_ALERTS_PER_INSTRUMENT", "15"))  # Max per instrument

# Choppy Market Detection
MIN_ATR_PERCENT = float(os.getenv("MIN_ATR_PERCENT", "0.3"))  # Min volatility for trading (ATR/price %)
MAX_VWAP_CROSSES = int(os.getenv("MAX_VWAP_CROSSES", "4"))  # Max crosses in 10 bars = choppy

# Correlation Limits
MAX_SAME_DIRECTION_ALERTS = int(os.getenv("MAX_SAME_DIRECTION_ALERTS", "3"))  # Max similar directional trades in 15 mins

SEND_TEST_ALERTS = os.getenv("SEND_TEST_ALERTS", "False").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "False").lower() == "true"
VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"


def validate_config():
    """Validate critical configuration"""
    errors = []

    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_KEY_HERE":
        errors.append("❌ GROQ_API_KEY not set in .env")

    if (
        not TELEGRAM_BOT_TOKEN
        or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE"
    ):
        errors.append("❌ TELEGRAM_BOT_TOKEN not set in .env")

    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        errors.append("❌ TELEGRAM_CHAT_ID not set in .env")

    return errors


os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if __name__ == "__main__":
    errs = validate_config()
    if errs:
        print("\n⚠️  Configuration Issues:\n")
        for e in errs:
            print(f"   {e}")
        print("\n📝 Please create .env file with required keys")
    else:
        print("✅ Configuration validated successfully!")
