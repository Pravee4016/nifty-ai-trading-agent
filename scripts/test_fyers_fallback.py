
import sys
import os
import logging
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_module.option_chain_fetcher import OptionChainFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fallback():
    fetcher = OptionChainFetcher()
    
    # Mock the NSE session.get to raise an exception
    with patch.object(fetcher.session, 'get', side_effect=Exception("Simulated NSE Failure")):
        
        # Mock the FyersApp to return dummy data so we don't need real credentials for this logic test
        with patch('data_module.fyers_interface.FyersApp') as MockFyersApp:
            mock_fyers_instance = MockFyersApp.return_value
            mock_fyers_instance.get_option_chain.return_value = {
                'data': {
                    'optionsChain': [
                        {'symbol': 'NSE:NIFTY23DEC19000CE', 'strike_price': 19000, 'oi': 100, 'volume': 500, 'ltp': 150},
                        {'symbol': 'NSE:NIFTY23DEC19000PE', 'strike_price': 19000, 'oi': 200, 'volume': 600, 'ltp': 20}
                    ],
                    'expiryData': [{'date': '28-Dec-2023'}]
                }
            }
            
            print("--- Testing Fallback Mechanism ---")
            data = fetcher.fetch_option_chain("NIFTY")
            
            if data:
                print("✅ Fallback successful!")
                print("Data keys:", data.keys())
                print("Records sample:", data['records']['data'][0])
            else:
                print("❌ Fallback failed")

if __name__ == "__main__":
    test_fallback()
