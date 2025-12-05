
import requests
import time
import logging
from typing import Dict, Optional

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
        self.cache_ttl = 60  # 1 minute cache
        
        # Initial visit to set cookies
        try:
            self.session.get("https://www.nseindia.com", timeout=10)
        except Exception as e:
            logger.warning(f"⚠️ Initial NSE visit failed: {e}")

    def _is_cache_valid(self, key: str) -> bool:
        if key in self.cache and key in self.cache_time:
            if time.time() - self.cache_time[key] < self.cache_ttl:
                return True
        return False

    def fetch_option_chain(self, instrument: str) -> Optional[Dict]:
        """
        Fetch option chain for NIFTY or BANKNIFTY.
        
        Args:
            instrument: Symbol name (e.g., "NIFTY", "BANKNIFTY")
            
        Returns:
            Dictionary containing option chain data or None if failed.
        """
        # Map instrument to NSE symbol format
        symbol = "NIFTY" if "NIFTY" in instrument and "BANK" not in instrument else "BANKNIFTY"
        if "FIN" in instrument: symbol = "FINNIFTY"
        
        # Check cache
        cache_key = f"oc_{symbol}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
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
            
            # Cache result
            self.cache[cache_key] = data
            self.cache_time[cache_key] = time.time()
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch option chain for {symbol}: {e}")
            return None

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
