
import os
import sys
from fyers_apiv3 import fyersModel

# Add project root to path
sys.path.append(os.getcwd())
from config.settings import FYERS_CLIENT_ID, FYERS_ACCESS_TOKEN

def check_token():
    print("Locked & Loaded: Checking Fyers Credentials...")
    
    if not FYERS_ACCESS_TOKEN:
        print("❌ ERROR: FYERS_ACCESS_TOKEN is not set in environment.")
        return

    print(f"Client ID: {FYERS_CLIENT_ID}")
    print(f"Token (First 10 chars): {FYERS_ACCESS_TOKEN[:10]}...")

    try:
        fyers = fyersModel.FyersModel(
            client_id=FYERS_CLIENT_ID,
            token=FYERS_ACCESS_TOKEN,
            is_async=False, 
            log_path=os.getcwd()
        )
        
        # Try to fetch profile/funds - a simple authenticated call
        print("\nAttempting to fetch Profile...")
        response = fyers.get_profile()
        
        print("\n--- RAW RESPONSE ---")
        print(response)
        
        if response.get("s") == "ok":
            print("\n✅ SUCCESS: Token is VALID.")
            print(f"User Name: {response.get('data', {}).get('name')}")
        else:
            print("\n❌ FAILURE: Token is INVALID or Expired.")
            print(f"Message: {response.get('message')}")

    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")

if __name__ == "__main__":
    check_token()
