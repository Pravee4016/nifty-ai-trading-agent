
import os
import sys
import logging
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()
from data_module.fyers_interface import FyersApp
logging.basicConfig(level=logging.ERROR) # Quiet logs

def verify_live_connection():
    print("--- Verifying Fyers Option Data ---")
    app_id = "DURQKS8D17-100" 
    fyers_app = FyersApp(app_id=app_id)
    
    if fyers_app.fyers:
        data = fyers_app.get_option_chain("NIFTY")
        if data and data.get("s") == "ok":
            chain = data.get('data', {}).get('optionsChain', [])
            
            # Find a real option
            option_contract = None
            for item in chain:
                if item.get('option_type') in ['CE', 'PE']:
                    option_contract = item
                    break
            
            if option_contract:
                print(f"✅ Found Option Contract: {option_contract.get('symbol')}")
                print(f"Keys: {list(option_contract.keys())}")
                print(f"OI: {option_contract.get('oi')}")
                print(f"Volume: {option_contract.get('volume')}")
                print(f"IV: {option_contract.get('iv')}") # Check if exists
                print("Full Item:")
                print(json.dumps(option_contract, indent=2))
            else:
                print("❌ No CE/PE contracts found in chain!")
        else:
            print("❌ Fetch failed")

if __name__ == "__main__":
    verify_live_connection()
