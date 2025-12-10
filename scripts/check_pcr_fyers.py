
import os
import sys
import logging
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from data_module.option_chain_fetcher import OptionChainFetcher
from analysis_module.option_chain_analyzer import OptionChainAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_pcr():
    print("--- Verifying PCR Calculation from Fyers ---")
    
    # 1. Fetch from Fyers (force fallback by using the direct method I added or just mocking)
    # Actually, OptionChainFetcher doesn't expose fetch_fyers_data publicly by default, 
    # but strictly speaking Python attributes are public.
    
    fetcher = OptionChainFetcher()
    analyzer = OptionChainAnalyzer()
    
    print("📡 Fetching data (forcing Fyers path)...")
    # We call the method we added directly to test that specific path
    data = fetcher.fetch_fyers_data("NIFTY")
    
    if data:
        print("✅ Data Fetched!")
        
        # 2. Calculate PCR
        pcr = analyzer.calculate_pcr(data)
        print(f"\n📊 Calculated PCR: {pcr}")
        
        # 3. Validation
        if pcr is not None and pcr > 0:
            print("✅ PCR Calculation Valid")
        else:
            print("❌ PCR Calculation Failed (Is None or 0)")
            
        # 4. Check details
        print("\n--- detailed debugging ---")
        records = data.get('records', {}).get('data', [])
        total_call_oi = sum(item.get('CE', {}).get('openInterest', 0) for item in records)
        total_put_oi = sum(item.get('PE', {}).get('openInterest', 0) for item in records)
        print(f"Total Call OI: {total_call_oi}")
        print(f"Total Put OI:  {total_put_oi}")
        if total_call_oi > 0:
             print(f"Manual Calc: {total_put_oi / total_call_oi:.4f}")
        
    else:
        print("❌ Failed to fetch data from Fyers")

if __name__ == "__main__":
    check_pcr()
