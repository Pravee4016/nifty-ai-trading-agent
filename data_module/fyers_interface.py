

import logging
import os
from typing import Dict, Optional, Any
from fyers_apiv3 import fyersModel
import time
import pandas as pd
from datetime import datetime, timedelta
import pytz

logger = logging.getLogger(__name__)

# Try to import OAuth manager (optional, graceful fallback)
try:
    from data_module.fyers_oauth import get_oauth_manager
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
    logger.debug("OAuth manager not available, using basic token auth")

class FyersApp:
    def __init__(self, app_id: str, secret_id: Optional[str] = None, access_token: Optional[str] = None):
        self.client_id = app_id  # App ID (e.g., DURQKS8D17-100)
        self.secret_key = secret_id
        self.access_token = access_token
        self.fyers: Optional[fyersModel.FyersModel] = None
        self.oauth_manager = None
        self.mapper = {
            "NIFTY": "NSE:NIFTY50-INDEX",
            "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
            "FINNIFTY": "NSE:FINNIFTY-INDEX",
            "NSE:INDIAVIX-INDEX": "NSE:INDIAVIX-INDEX"
        }
        
        # Try to use OAuth if available
        if OAUTH_AVAILABLE and secret_id:
            try:
                self.oauth_manager = get_oauth_manager()
                logger.info("🔐 Using OAuth manager for automatic token refresh")
            except Exception as e:
                logger.debug(f"OAuth manager not initialized: {e}")
        
        self.initialize_session()

    def initialize_session(self):
        """
        Initialize Fyers Model.
        Requires access_token. If not provided, it attempts to read from env or OAuth.
        """
        # Try OAuth first if available
        if self.oauth_manager and self.oauth_manager.is_authorized():
            self.access_token = self.oauth_manager.get_valid_access_token()
            if self.access_token:
                logger.info("✅ Using OAuth-managed access token")
        
        # Fallback to environment variable
        if not self.access_token:
            # Try to load from env
            self.access_token = os.getenv("FYERS_ACCESS_TOKEN")
        
        if not self.access_token:
            # If no token, we can't make authenticated calls
            logger.warning("⚠️ Fyers Access Token is missing. Fyers API calls will fail.")
            return

        try:
            self.fyers = fyersModel.FyersModel(
                client_id=self.client_id,
                token=self.access_token,
                is_async=False, 
                log_path=os.getcwd()
            )
            logger.info("✅ Fyers Model initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Fyers Model: {e}")

    def validate_session(self) -> bool:
        """
        Validate if the current Fyers session is active.
        Returns True if session is valid, False otherwise.
        """
        if not self.fyers:
            return False
        
        try:
            # Try a simple API call to check token validity
            test_data = {"symbols": "NSE:NIFTY50-INDEX"}
            response = self.fyers.quotes(data=test_data)
            
            if response.get("s") == "ok":
                logger.debug("✅ Fyers session is valid")
                return True
            else:
                error_msg = response.get('message', 'Unknown')
                if 'token' in error_msg.lower() or 'invalid' in error_msg.lower():
                    logger.warning(f"⚠️ Fyers token expired or invalid: {error_msg}")
                    return False
                return False
        except Exception as e:
            logger.warning(f"⚠️ Fyers session validation failed: {e}")
            return False
    
    def refresh_token_from_env(self) -> bool:
        """
        Attempt to refresh token from environment variable.
        Useful if token is updated externally.
        Returns True if new token loaded successfully.
        """
        new_token = os.getenv("FYERS_ACCESS_TOKEN")
        if new_token and new_token != self.access_token:
            logger.info("🔄 Attempting to refresh Fyers token from environment")
            self.access_token = new_token
            self.initialize_session()
            return self.validate_session()
        return False

    def get_option_chain(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch option chain for the given symbol.
        Args:
            symbol: Common symbol name (e.g., "NIFTY", "BANKNIFTY")
        """
        if not self.fyers:
            logger.error("❌ Fyers session not initialized")
            return None
            
        fyers_symbol = self.mapper.get(symbol)
        if not fyers_symbol:
            logger.error(f"❌ Unknown symbol for Fyers: {symbol}")
            return None
            
        try:
            # According to Fyers API v3 docs
            data = {
                "symbol": fyers_symbol,
                "strikecount": 20, # Fetch adequate strikes
                "timestamp": ""
            }
            
            response = self.fyers.optionchain(data=data)
            
            if response.get("s") == "ok":
                logger.info(f"✅ Fyers Option Chain fetched for {symbol}")
                return response
            else:
                logger.error(f"❌ Fyers API Error: {response.get('message', 'Unknown Error')}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Exception fetching Fyers Option Chain: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch Level 1 quote (LTP, OHLC) for a symbol.
        """
        if not self.fyers:
            # Try re-initializing if token is available
            self.initialize_session()
            if not self.fyers:
                logger.debug("Fyers not available, caller will use fallback")
                return None
        
        # Validate session before making call
        if not self.validate_session():
            logger.warning("⚠️ Fyers session invalid, trying token refresh...")
            if not self.refresh_token_from_env():
                logger.info("ℹ️ Fyers unavailable - system will use yfinance fallback")
                return None
                
        fyers_symbol = self.mapper.get(symbol)
        if not fyers_symbol:
            logger.error(f"❌ Unknown symbol: {symbol}")
            return None

        try:
            data = {"symbols": fyers_symbol}
            response = self.fyers.quotes(data=data)
            
            if response.get("s") == "ok" and "d" in response:
                return response["d"][0] # Return the first (and only) result
            else:
                logger.error(f"❌ Fyers Quote Failed: {response.get('message')}")
                return None
        except Exception as e:
            logger.error(f"❌ Exception fetching quote: {e}")
            return None

    def get_historical_candles(
        self,
        symbol: str,
        resolution: str = "1",
        bars: int = 10
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical candles from Fyers API.
        
        Args:
            symbol: Common symbol name (e.g., "NIFTY", "BANKNIFTY")
            resolution: "1" (1m), "5" (5m), "15" (15m), "30" (30m), "60" (1h), "D" (daily)
            bars: Number of candles to fetch
        
        Returns:
            DataFrame with columns: ['datetime', 'open', 'high', 'low', 'close', 'volume']
            or None if fetch fails
        """
        if not self.fyers:
            self.initialize_session()
            if not self.fyers:
                logger.debug("Fyers not available, caller should use yfinance fallback")
                return None
        
        # Validate session
        if not self.validate_session():
            logger.warning("⚠️ Fyers session invalid for history fetch")
            if not self.refresh_token_from_env():
                return None
        
        fyers_symbol = self.mapper.get(symbol)
        if not fyers_symbol:
            logger.error(f"❌ Unknown symbol for Fyers: {symbol}")
            return None
        
        try:
            # Calculate date range based on resolution and bars
            ist = pytz.timezone('Asia/Kolkata')
            end_time = datetime.now(ist)
            
            # Calculate start time based on resolution and bars needed
            resolution_minutes = {
                "1": 1,
                "5": 5,
                "15": 15,
                "30": 30,
                "60": 60,
                "D": 1440  # Daily
            }
            
            minutes_needed = resolution_minutes.get(resolution, 5) * bars
            
            # For intraday data (1m, 5m, 15m, etc.), use day-based range
            # to account for market hours (09:15-15:30 IST = ~6 hours)
            if resolution in ["1", "5", "15", "30", "60"]:
                # Use 2-3 days lookback for intraday to ensure enough candles
                start_time = end_time - timedelta(days=3)
            else:
                # For daily data
                start_time = end_time - timedelta(days=bars * 2)
            
            # Fyers API requires date strings in YYYY-MM-DD format when date_format=1
            range_from = start_time.strftime("%Y-%m-%d")
            range_to = end_time.strftime("%Y-%m-%d")
            
            data = {
                "symbol": fyers_symbol,
                "resolution": resolution,
                "date_format": "1",  # Date format: YYYY-MM-DD
                "range_from": range_from,
                "range_to": range_to
                # cont_flag removed - may cause "Invalid input" for intraday
            }
            
            # Debug: Log API call details
            logger.info(f"📊 Fyers API Call | Symbol: {symbol} → {fyers_symbol} | Resolution: {resolution} | Bars: {bars}")
            logger.info(f"   Date Range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"   Timestamps: {range_from} to {range_to}")
            logger.info(f"   API Data: {data}")
            
            response = self.fyers.history(data=data)
            
            if response.get("s") != "ok":
                logger.error(f"❌ Fyers history API error: {response.get('message', 'Unknown')}")
                logger.error(f"   Full response: {response}")
                return None
            
            # Parse response
            candles = response.get("candles", [])
            
            if not candles:
                logger.warning(f"⚠️ No candles returned from Fyers for {symbol}")
                return None
            
            # Convert to DataFrame
            # Fyers candles format: [timestamp, open, high, low, close, volume]
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime (IST)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df['datetime'] = df['datetime'].dt.tz_convert(ist)
            
            # Drop timestamp column, reorder
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            # Take only the requested number of bars (most recent)
            df = df.tail(bars)
            
            logger.info(f"✅ Fetched {len(df)} {symbol} candles from Fyers (resolution: {resolution})")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Exception fetching Fyers historical data: {e}")
            return None


if __name__ == "__main__":
    # Test Block
    APP_ID = "DURQKS8D17-100"
    SECRET_ID = os.getenv("FYERS_SECRET_ID") # User said they have it
    ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
    
    app = FyersApp(app_id=APP_ID, secret_id=SECRET_ID, access_token=ACCESS_TOKEN)
    if app.fyers:
        print("Testing Fetch...")
        data = app.get_option_chain("NIFTY")
        print(data)
