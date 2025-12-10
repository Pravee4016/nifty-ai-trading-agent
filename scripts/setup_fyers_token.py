
import os
import sys
from fyers_apiv3 import fyersModel

def setup_token():
    print("--- Fyers Access Token Setup ---")
    
    app_id = input("Enter Fyers App ID (Client ID) [Default: DURQKS8D17-100]: ").strip() or "DURQKS8D17-100"
    secret_id = input("Enter Fyers Secret ID: ").strip()
    redirect_uri = input("Enter Redirect URI [Default: https://trade.fyers.in/api-login/redirect-uri/index.html]: ").strip() or "https://trade.fyers.in/api-login/redirect-uri/index.html"
    
    if not secret_id:
        print("❌ Secret ID is required!")
        return

    # 1. Create session to get auth URL
    session = fyersModel.SessionModel(
        client_id=app_id,
        secret_key=secret_id,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code"
    )


    auth_link = session.generate_authcode()
    print("\n🔗 Open this URL in your browser and login:")
    print(auth_link)
    print("\n")
    
    print("1. Login to Fyers with the link above.")
    print("2. After login, you will be redirected to a URL.")
    print("3. Copy that ENTIRE URL from your browser address bar and paste it below.")
    
    url_or_code = input("\nPaste the Full Redirect URL here: ").strip()
    
    # helper to extract code
    from urllib.parse import parse_qs, urlparse
    
    if "auth_code=" in url_or_code:
        # It's likely a URL
        try:
            parsed = urlparse(url_or_code)
            params = parse_qs(parsed.query)
            auth_code = params.get('auth_code', [None])[0]
        except:
            auth_code = url_or_code # Fallback
    else:
        # Assume it's the code directly
        auth_code = url_or_code
    
    if not auth_code:
        print("❌ Auth code is required!")
        return

    # 2. Generate Access Token
    session.set_token(auth_code)
    response = session.generate_token()
    
    if response.get("code") == 200:
        access_token = response["access_token"]
        print("\n✅ Access Token Generated Successfully!")
        print(f"Access Token: {access_token}")
        print("\n📝 Add this to your .env file or environment variables:")
        print(f"FYERS_ACCESS_TOKEN={access_token}")
        print(f"FYERS_SECRET_ID={secret_id}")
        
        # Optional: Append to .env automatically
        save = input("\nDo you want to append this to .env? (y/n): ").lower()
        if save == 'y':
            with open(".env", "a") as f:
                f.write(f"\n# Fyers API\nFYERS_ACCESS_TOKEN={access_token}\nFYERS_SECRET_ID={secret_id}\n")
            print("💾 Saved to .env")
    else:
        print(f"❌ Failed to generate token: {response}")

if __name__ == "__main__":
    setup_token()

