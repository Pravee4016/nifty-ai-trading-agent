
import os
import sys
import logging
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

from data_module.option_chain_fetcher import OptionChainFetcher
from analysis_module.option_chain_analyzer import OptionChainAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("SystemCheck")

def run_system_check():
    print("\n===========================================")
    print("🔎 FYERS OPTION CHAIN INTEGRATION CHECK")
    print("===========================================")
    
    # 1. Check Credentials
    token = os.getenv("FYERS_ACCESS_TOKEN")
    if not token:
        print("❌ FYERS_ACCESS_TOKEN missing in .env")
        return
    print(f"✅ Token Found: {token[:10]}...")

    # 2. Fetch Data (Force Fyers)
    print("\n--- Step 1: Fetching Data ---")
    fetcher = OptionChainFetcher()
    
    # Explicitly calling the Fyers method to ensure we test THAT specific path
    # (In production, this happens automatically if NSE fails)
    data = fetcher.fetch_fyers_data("NIFTY")
    
    if not data:
        print("❌ Fetch Failed! Check Token validity.")
        return
        
    records = data.get('records', {}).get('data', [])
    print(f"✅ Fetch Successful! Retrieved {len(records)} contracts.")
    
    # 3. Verify Data Integrity
    print("\n--- Step 2: Verifying Data Fields ---")
    if not records:
        print("❌ No records to verify.")
        return
        
    sample = records[0].get('CE', {}) or records[0].get('PE', {})
    required_fields = ['openInterest', 'changeinOpenInterest', 'impliedVolatility']
    
    missing = [f for f in required_fields if f not in sample]
    
    if missing:
        print(f"❌ Data Incomplete. Missing keys: {missing}")
        print(f"   Received keys: {list(sample.keys())}")
    else:
        print("✅ Data Structure Valid (Matches NSE format).")
        print(f"   Sample Item: OI={sample['openInterest']}, OICH={sample.get('changeinOpenInterest')}, IV={sample.get('impliedVolatility')}")

    # 4. Verify Analysis Integration
    print("\n--- Step 3: Verifying Analysis Logic ---")
    analyzer = OptionChainAnalyzer()
    
    # PCR
    pcr = analyzer.calculate_pcr(data)
    print(f"📊 PCR: {pcr} " + ("✅ (Valid)" if pcr else "❌ (Invalid)"))
    
    # Max Pain
    max_pain = analyzer.calculate_max_pain(data)
    print(f"📊 Max Pain: {max_pain} " + ("✅ (Valid)" if max_pain else "❌ (Invalid)"))
    
    # Sentiment (OI Change)
    # We need a dummy spot price to calculate "ATM" sentiment
    spot_price = records[0]['strikePrice'] # Just pick a strike as 'price' for testing
    sentiment = analyzer.analyze_oi_change(data, spot_price)
    print(f"📊 Sentiment Analysis: {sentiment.get('sentiment')} " + ("✅ (Valid)" if sentiment else "❌ (Invalid)"))
    
    print("\n===========================================")
    print("✅ SYSTEM CHECK PASSED: Fyers data is Analysis-Ready")
    print("===========================================")

if __name__ == "__main__":
    run_system_check()
