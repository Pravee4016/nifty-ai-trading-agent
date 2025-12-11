
import logging
import os
from typing import Dict, Optional, Any
from fyers_apiv3 import fyersModel
import time

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
            "FINNIFTY": "NSE:FINNIFTY-INDEX"
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
