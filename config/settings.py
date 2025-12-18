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
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID") # Optional Channel ID

# ============================================================================
# FYERS CONFIGURATION
# ============================================================================
FYERS_CLIENT_ID = os.getenv("FYERS_CLIENT_ID", "DURQKS8D17-100") # Default from codebase
FYERS_SECRET_ID = os.getenv("FYERS_SECRET_ID")  # For OAuth refresh token support
FYERS_ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")


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
        "tick_size": 0.05,  # Legacy - use get_tick_size() instead
        "lot_size": 75,
        "active": True,
    },
    "BANKNIFTY": {
        "symbol": "BANKNIFTY",
        "display_name": "Bank Nifty",
        "tick_size": 0.05,  # Legacy - use get_tick_size() instead
        "lot_size": 15,
        "active": True,
    },
    "FINNIFTY": {
        "symbol": "FINNIFTY",
        "display_name": "Fin Nifty",
        "tick_size": 0.05,  # Legacy - use get_tick_size() instead
        "lot_size": 40,
        "active": False,  # Enable when ready
    },
}

# ============================================================================
# TICK SIZE MAPPING (Spot vs Options)
# ============================================================================

# CRITICAL: Spot instruments trade in 1-point increments, NOT 0.05
# 0.05 tick size is ONLY for options (CE/PE)
TICK_SIZE_MAP = {
    "NIFTY_SPOT": 1.0,           # NIFTY spot/futures move in 1.0 point increments
    "NIFTY_OPTION": 0.05,        # NIFTY options (CE/PE) trade in 0.05 ticks
    "BANKNIFTY_SPOT": 1.0,       # BankNifty spot/futures
    "BANKNIFTY_OPTION": 0.05,    # BankNifty options
    "FINNIFTY_SPOT": 1.0,        # FinNifty spot/futures
    "FINNIFTY_OPTION": 0.05,     # FinNifty options
}


def get_tick_size(instrument: str, is_option: bool = False) -> float:
    """
    Get correct tick size for instrument and product type.
    
    Args:
        instrument: Instrument name (e.g., "NIFTY", "BANKNIFTY", "FINNIFTY")
        is_option: True if trading options (CE/PE), False for spot/futures
    
    Returns:
        Tick size (1.0 for spot, 0.05 for options)
    
    Example:
        >>> get_tick_size("NIFTY", is_option=False)  # Spot/Futures
        1.0
        >>> get_tick_size("NIFTY", is_option=True)   # Options
        0.05
    """
    base = instrument.upper()
    
    # Normalize instrument name
    if "BANK" in base:
        key = "BANKNIFTY"
    elif "FIN" in base:
        key = "FINNIFTY"
    else:
        key = "NIFTY"
    
    suffix = "_OPTION" if is_option else "_SPOT"
    tick_size = TICK_SIZE_MAP.get(f"{key}{suffix}", 1.0)
    
    return tick_size

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
RETEST_ZONE_PERCENT = float(os.getenv("RETEST_ZONE_PERCENT", 0.15))

# Support/Resistance Configuration
SR_CLUSTER_TOLERANCE = 0.1  # 0.1% difference = same cluster
MIN_SR_TOUCHES = 2
LOOKBACK_BARS = 2000  # Increased to use full 5-day history (was 100)

# Momentum Configuration
MIN_RSI_BULLISH = int(os.getenv("MIN_MOMENTUM_RSI", 50))  # Lowered from 60 to catch early trends
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
MAX_DAILY_TRADES = 999  # Effectively unlimited (rely on robust filtering)
MIN_SIGNAL_CONFIDENCE = int(os.getenv("MIN_SIGNAL_CONFIDENCE", 65))
MIN_SCORE_THRESHOLD = int(os.getenv("MIN_SCORE_THRESHOLD", 65))  # Raised from 60 for expert-level quality

# Feature Flag: Expert Analysis Enhancements
# Set to False to rollback to previous scoring without code changes
USE_EXPERT_ENHANCEMENTS = os.getenv("USE_EXPERT_ENHANCEMENTS", "True").lower() == "true"

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
# VERTEX AI CONFIGURATION (Gemini)
# ============================================================================

# AI Provider Selection: GROQ, VERTEX, or HYBRID (A/B testing)
AI_PROVIDER = os.getenv("AI_PROVIDER", "GROQ")  # Options: GROQ, VERTEX, HYBRID

# Vertex AI Settings
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "nifty-trading-agent"))
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.0-flash-exp")

# Hybrid Mode Configuration (when AI_PROVIDER=HYBRID)
HYBRID_GROQ_WEIGHT = float(os.getenv("HYBRID_GROQ_WEIGHT", 0.5))    # 50% to Groq
HYBRID_VERTEX_WEIGHT = float(os.getenv("HYBRID_VERTEX_WEIGHT", 0.5))  # 50% to Vertex

# ============================================================================
# MACHINE LEARNING CONFIGURATION
# ============================================================================

# Enable/Disable ML Filtering
USE_ML_FILTERING = os.getenv("USE_ML_FILTERING", "False").lower() == "true"

# Google Cloud Storage for Models
ML_MODEL_BUCKET = os.getenv("ML_MODEL_BUCKET", "nifty-trading-agent-ml-models")
ML_MODEL_NAME = os.getenv("ML_MODEL_NAME", "signal_quality_v1.txt")

# ML Prediction Thresholds
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", 0.65))  # Min probability to accept signal
ML_FALLBACK_TO_RULES = bool(os.getenv("ML_FALLBACK_TO_RULES", "True"))  # Use rule-based if ML unavailable

# Retraining Configuration  
ML_RETRAIN_FREQUENCY_DAYS = int(os.getenv("ML_RETRAIN_FREQUENCY", 7))  # Weekly
ML_MIN_TRAINING_SAMPLES = int(os.getenv("ML_MIN_TRAINING_SAMPLES", 100))  # Minimum trades to train

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
# ============================================================================
# CLOUD DEPLOYMENT
# ============================================================================

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
CLOUD_FUNCTION_REGION = os.getenv("CLOUD_FUNCTION_REGION", "us-central1")
FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE", "")

# Auto-detect GCP Environment (Cloud Run / Cloud Functions)
IS_GCP = os.getenv("K_SERVICE") is not None or os.getenv("FUNCTION_NAME") is not None

# Override DEPLOYMENT_MODE if in GCP
if IS_GCP:
    DEPLOYMENT_MODE = "GCP"
else:
    DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "LOCAL")

ENABLE_CLOUD_LOGGING = DEPLOYMENT_MODE != "LOCAL"

# ============================================================================
# DATA STORAGE & CACHING
# ============================================================================

if DEPLOYMENT_MODE == "GCP" or IS_GCP:
    # Google Cloud Functions only allows writing to /tmp
    CACHE_DIR = "/tmp/cache"
    LOG_DIR = "/tmp/logs"
else:
    CACHE_DIR = "./cache"
    LOG_DIR = "./logs"

CACHE_DATA_TTL = 3600  # seconds

LOG_LEVEL = os.getenv("LOGLEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

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
MIN_ATR_PERCENT = float(os.getenv("MIN_ATR_PERCENT", "0.06"))  # Min volatility for trading (ATR/price %)
MAX_VWAP_CROSSES = int(os.getenv("MAX_VWAP_CROSSES", "4"))  # Max crosses in 10 bars = choppy

# Correlation Limits
MAX_SAME_DIRECTION_ALERTS = int(os.getenv("MAX_SAME_DIRECTION_ALERTS", "3"))  # Max similar directional trades in 15 mins

SEND_TEST_ALERTS = os.getenv("SEND_TEST_ALERTS", "False").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "False").lower() == "true"
VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"

# ============================================================================
# PHASE 4: MANIPULATION DEFENSE
# ============================================================================

# Flash Crash Protection
MAX_1MIN_MOVE_PCT = float(os.getenv("MAX_1MIN_MOVE_PCT", 0.4))  # 0.4% move in 1 min = Freeze
CIRCUIT_BREAKER_PAUSE_MINS = 15

# Expiry Day Gamma Guard ( Thursdays )
EXPIRY_STOP_TIME = "14:00"  # Stop taking fresh entries after this time on Thursdays
EXPIRY_RISK_ATR_MULTIPLIER = 1.0  # Tighter stops on Expiry

# VIX Volatility Guard
VIX_PANIC_LEVEL = 24.0
VIX_LOW_LEVEL = 10.0


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
