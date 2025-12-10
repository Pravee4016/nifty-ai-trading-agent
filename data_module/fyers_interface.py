
import logging
import os
from typing import Dict, Optional, Any
from fyers_apiv3 import fyersModel
import time

logger = logging.getLogger(__name__)

class FyersApp:
    def __init__(self, app_id: str, secret_id: Optional[str] = None, access_token: Optional[str] = None):
        self.client_id = app_id  # App ID (e.g., DURQKS8D17-100)
        self.secret_key = secret_id
        self.access_token = access_token
        self.fyers: Optional[fyersModel.FyersModel] = None
        self.mapper = {
            "NIFTY": "NSE:NIFTY50-INDEX",
            "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
            "FINNIFTY": "NSE:FINNIFTY-INDEX"
        }
        
        self.initialize_session()

    def initialize_session(self):
        """
        Initialize Fyers Model.
        Requires access_token. If not provided, it attempts to read from env or file.
        """
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
