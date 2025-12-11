#!/usr/bin/env python3
"""
Fyers OAuth Setup Script
One-time setup to authorize the app and obtain refresh token
"""

import os
import sys
import webbrowser
from urllib.parse import parse_qs, urlparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_module.fyers_oauth import get_oauth_manager
from dotenv import load_dotenv

load_dotenv()


def main():
    print("=" * 60)
    print("🔐 Fyers OAuth Setup - One-Time Authorization")
    print("=" * 60)
    print()
    
    # Get OAuth manager
    try:
        oauth_manager = get_oauth_manager()
    except ValueError as e:
        print(f"❌ Error: {e}")
        print()
        print("Please ensure the following environment variables are set:")
        print("  - FYERS_CLIENT_ID")
        print("  - FYERS_SECRET_ID")
        return 1
    
    # Check if already authorized
    if oauth_manager.is_authorized():
        print("✅ Already authorized! Refresh token found.")
        print()
        print("Testing token refresh...")
        success, message = oauth_manager.refresh_access_token()
        if success:
            print(f"✅ {message}")
            print(f"📝 Access Token: {oauth_manager.access_token[:20]}...")
            print()
            print("🎉 OAuth is working! Your system will auto-refresh tokens.")
            return 0
        else:
            print(f"❌ Refresh failed: {message}")
            print("❌ Need to re-authorize. Continuing with setup...")
            print()
    
    # Generate authorization URL
    print("Step 1: Authorize the Application")
    print("-" * 60)
    auth_url = oauth_manager.generate_auth_url()
    print()
    print("Opening browser for authorization...")
    print("If browser doesn't open, visit this URL:")
    print(auth_url)
    print()
    
    # Open browser
    try:
        webbrowser.open(auth_url)
    except:
        print("⚠️ Could not open browser automatically")
    
    print()
    print("Step 2: Get Authorization Code")
    print("-" * 60)
    print("After authorizing:")
    print("  1. You'll be redirected to a URL")
    print("  2. Copy the ENTIRE redirect URL")
    print("  3. Paste it below")
    print()
    
    # Get redirect URL from user
    redirect_url = input("Paste redirect URL here: ").strip()
    
    # Extract auth code
    try:
        parsed = urlparse(redirect_url)
        params = parse_qs(parsed.query)
        auth_code = params.get('auth_code', [None])[0]
        
        if not auth_code:
            print("❌ Could not find auth_code in URL")
            print("Make sure you copied the complete redirect URL")
            return 1
        
        print()
        print("Step 3: Exchanging Code for Tokens")
        print("-" * 60)
        print(f"📝 Auth Code: {auth_code[:20]}...")
        
        success, message = oauth_manager.exchange_auth_code(auth_code)
        
        if success:
            print()
            print("=" * 60)
            print("🎉 SUCCESS! OAuth Setup Complete")
            print("=" * 60)
            print()
            print("✅ Access token obtained")
            print("✅ Refresh token obtained and saved to Cloud Secret Manager")
            print()
            print("Your system will now automatically:")
            print("  - Refresh access tokens before expiry")
            print("  - Never require manual token updates")
            print("  - Persist across deployments")
            print()
            print("=" * 60)
            print("Next Steps:")
            print("  1. The refresh token is already in Cloud Secret Manager")
            print("  2. Redeploy your application: ./deploy_job.sh")
            print("  3. Tokens will auto-refresh indefinitely!")
            print("=" * 60)
            
            # Optionally update .env with new access token
            update_env = input("\nUpdate .env with new access token? (y/n): ").lower()
            if update_env == 'y':
                env_file = ".env"
                if os.path.exists(env_file):
                    # Read existing .env
                    with open(env_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Update or add FYERS_ACCESS_TOKEN
                    updated = False
                    for i, line in enumerate(lines):
                        if line.startswith('FYERS_ACCESS_TOKEN='):
                            lines[i] = f'FYERS_ACCESS_TOKEN={oauth_manager.access_token}\n'
                            updated = True
                            break
                    
                    if not updated:
                        lines.append(f'\nFYERS_ACCESS_TOKEN={oauth_manager.access_token}\n')
                    
                    # Write back
                    with open(env_file, 'w') as f:
                        f.writelines(lines)
                    
                    print(f"✅ Updated {env_file}")
                else:
                    print(f"⚠️ {env_file} not found, skipping")
            
            return 0
        else:
            print()
            print(f"❌ Failed to exchange auth code: {message}")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
