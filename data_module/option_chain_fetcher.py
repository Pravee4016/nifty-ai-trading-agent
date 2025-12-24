
import requests
import time
import logging
from typing import Dict, Optional
from datetime import datetime
from config.settings import FYERS_CLIENT_ID
from data_module.fyers_interface import FyersApp

logger = logging.getLogger(__name__)

class OptionChainFetcher:
    """
    Fetches option chain data from NSE official API.
    Handles session management and headers to mimic a browser.
    """
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/option-chain'
        })
        self.cache = {}
        self.cache_time = {}
        self.cache_ttl = 300  # 5 minutes (increased from 60 for production stability)
        
        # NEW: Emergency cache for fallback when both Fyers AND NSE fail
        self.last_valid_data = {}
        self.degraded_mode = False
        
        # Initial visit to set cookies
        try:
            self.session.get("https://www.nseindia.com", timeout=10)
        except Exception as e:
            logger.warning(f"⚠️ Initial NSE visit failed: {e}")
            
        # Initialize Fyers App for PRIMARY data source
        self.fyers_app = FyersApp(app_id=FYERS_CLIENT_ID)

    def _is_cache_valid(self, key: str) -> bool:
        if key in self.cache and key in self.cache_time:
            if time.time() - self.cache_time[key] < self.cache_ttl:
                return True
        return False

    def fetch_option_chain(self, instrument: str) -> Optional[Dict]:
        """
        Fetch option chain for NIFTY or BANKNIFTY.
        Prioritizes Fyers API. Falls back to NSE website if Fyers fails.
        
        Args:
            instrument: Symbol name (e.g., "NIFTY", "BANKNIFTY")
            
        Returns:
            Dictionary containing option chain data or None if failed.
        """
        # Map instrument to NSE symbol format
        symbol = "NIFTY" if "NIFTY" in instrument and "BANK" not in instrument else "BANKNIFTY"
        if "FIN" in instrument: symbol = "FINNIFTY"
        
        cache_key = f"oc_{symbol}"
        if self._is_cache_valid(cache_key):
            self.degraded_mode = False  # Reset if cache is working
            return self.cache[cache_key]

        # ----------------------------------------
        # PRIMARY: FYERS API
        # ----------------------------------------
        data = self.fetch_fyers_data(instrument)
        if data:
            # CRITICAL: Add timestamp for staleness validation
            data['fetch_timestamp'] = time.time()
            data['fetch_age_seconds'] = 0
            
            self.cache[cache_key] = data
            self.cache_time[cache_key] = time.time()
            # Store as emergency backup
            self.last_valid_data[cache_key] = data
            self.degraded_mode = False
            return data

        # ----------------------------------------
        # FALLBACK: NSE WEBSITE (Scraping)
        # ----------------------------------------
        logger.warning(f"⚠️ Fyers Option Chain failed. Falling back to NSE Scraping for {symbol}")
        
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        
        try:
            # Add specific headers for API call
            headers = {
                'Referer': f'https://www.nseindia.com/option-chain?symbol={symbol}'
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 401:
                # Refresh session if unauthorized
                logger.info("🔄 Refreshing NSE session...")
                self.session.get("https://www.nseindia.com", timeout=10)
                response = self.session.get(url, headers=headers, timeout=10)

            response.raise_for_status()
            data = response.json()
            
            # Validate Data
            if "records" not in data or "data" not in data.get("records", {}):
                raise ValueError("Invalid NSE data structure (missing records)")

            # CRITICAL: Add timestamp for staleness validation
            data['fetch_timestamp'] = time.time()
            data['fetch_age_seconds'] = 0

            # Cache result
            self.cache[cache_key] = data
            self.cache_time[cache_key] = time.time()
            # Store as emergency backup
            self.last_valid_data[cache_key] = data
            self.degraded_mode = False
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch option chain for {symbol}: {e}")
            
            # ----------------------------------------
            # EMERGENCY FALLBACK: Use last known good data
            # ----------------------------------------
            if cache_key in self.last_valid_data:
                logger.warning(f"⚠️ Using STALE option chain data (>5min old) for {symbol}")
                self.degraded_mode = True
                return self.last_valid_data[cache_key]
            
            logger.critical(f"❌ NO OPTION CHAIN DATA AVAILABLE for {symbol}")
            self.degraded_mode = True
            return None
    
    def is_healthy(self) -> bool:
        """Check if option chain fetcher is working normally."""
        return not self.degraded_mode

    def fetch_fyers_data(self, instrument: str) -> Optional[Dict]:
        """Fetch and transform data from Fyers."""
        try:
            raw_data = self.fyers_app.get_option_chain(instrument)
            if raw_data and raw_data.get('data'):
                return self._transform_fyers_to_nse(raw_data['data'])
            return None
        except Exception as e:
            logger.error(f"❌ Fyers Fallback failed: {e}")
            return None

    def _transform_fyers_to_nse(self, fyers_data: Dict) -> Dict:
        """
        Transform Fyers response to match NSE structure for compatibility.
        """
        options_chain = fyers_data.get('optionsChain', [])
        expiry_data = fyers_data.get('expiryData', [])
        
        grouped = {}
        for item in options_chain:
            strike = item.get('strike_price')
            if not strike or strike <= 0: continue
            
            if strike not in grouped:
                grouped[strike] = {'strikePrice': strike}
            
            # Determine type
            sym = item.get('symbol', '')
            if 'CE' in sym[-2:]: type_key = 'CE'
            elif 'PE' in sym[-2:]: type_key = 'PE'
            else: continue
            
            # Map fields
            # Note: Fyers keys need to be verified. Assuming standard keys here.
            # Adjust keys based on actual Fyers API response inspection if needed.
            node = {
                'strikePrice': strike,
                'openInterest': item.get('oi', 0),
                'changeinOpenInterest': item.get('oich', 0),
                'totalTradedVolume': item.get('volume', 0),
                'impliedVolatility': item.get('iv', 0),
                'lastPrice': item.get('ltp', 0),
                'change': item.get('ltpch', 0), 
                'pChange': item.get('ltpchp', 0)
            }
            grouped[strike][type_key] = node

        unique_expiries = [x.get('date') for x in expiry_data if x.get('date')]
        
        data_list = sorted(list(grouped.values()), key=lambda x: x['strikePrice'])
        
        return {
            'records': {
                'expiryDates': unique_expiries,
                'data': data_list,
                'timestamp':  datetime.now().strftime("%d-%b-%Y %H:%M:%S")
            },
            'filtered': {
                'data': data_list, # Providing full list as filtered for now
                'CE': {'totOI': 0, 'totVol': 0}, 
                'PE': {'totOI': 0, 'totVol': 0}
            }
        }

if __name__ == "__main__":
    # Test execution
    logging.basicConfig(level=logging.INFO)
    fetcher = OptionChainFetcher()
    data = fetcher.fetch_option_chain("NIFTY")
    if data:
        print("✅ Fetch success")
        records = data.get('records', {})
        print(f"Expiry Dates: {records.get('expiryDates', [])[:3]}")
    else:
        print("❌ Fetch failed")
