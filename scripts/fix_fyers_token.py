#!/usr/bin/env python3
"""
Fyers Token Generator - With PIN
"""

import requests
import json
import sys
import subprocess

print("=" * 70)
print("🔑 FYERS TOKEN GENERATOR (WITH PIN)")
print("=" * 70)

# Get refresh token and PIN from Secret Manager
print("\n📥 Getting credentials from Secret Manager...")

# Get refresh token
result = subprocess.run(
    ['gcloud', 'secrets', 'versions', 'access', 'latest', 
     '--secret=fyers-refresh-token', '--project=nifty-trading-agent'],
    capture_output=True, text=True
)
refresh_token = result.stdout.strip()

# Get PIN
result = subprocess.run(
    ['gcloud', 'secrets', 'versions', 'access', 'latest', 
     '--secret=fyers-pin', '--project=nifty-trading-agent'],
    capture_output=True, text=True
)
pin = result.stdout.strip()

print(f"✅ Refresh token: {refresh_token[:50]}...")
print(f"✅ PIN: {'*' * len(pin)}")

# Credentials
app_id = "DURQKS8D17-100"
secret_key = "9ZWWXGKO4C"

print(f"\nApp ID: {app_id}")
print(f"Secret Key: {secret_key[:10]}...")

# Generate hash
import hashlib
app_hash = hashlib.sha256(f"{app_id}:{secret_key}".encode()).hexdigest()

# Make API call with PIN
print("\n🔄 Generating new access token...")

url = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
headers = {"Content-Type": "application/json"}
payload = {
    "grant_type": "refresh_token",
    "appIdHash": app_hash,
    "refresh_token": refresh_token,
    "pin": pin
}

try:
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    if response.status_code == 200 and 'access_token' in data:
        access_token = data['access_token']
        
        print("\n" + "=" * 70)
        print("✅ NEW ACCESS TOKEN GENERATED!")
        print("=" * 70)
        print(access_token)
        print("=" * 70)
        print(f"Length: {len(access_token)}")
        
        # Update Secret Manager
        print("\n📤 Updating Secret Manager...")
        result = subprocess.run(
            ['gcloud', 'secrets', 'versions', 'add', 'fyers-access-token',
             '--data-file=-', '--project=nifty-trading-agent'],
            input=access_token.encode(),
            capture_output=True
        )
        
        if result.returncode == 0:
            print("✅ Secret Manager updated!")
            print("\n" + "=" * 70)
            print("✅ DONE! Token is now active in production")
            print("=" * 70)
            print("\nNext job run will use the new token")
        else:
            print("❌ Failed to update Secret Manager")
            print("Manual update command:")
            print(f"\necho '{access_token}' | gcloud secrets versions add fyers-access-token --data-file=- --project=nifty-trading-agent")
    else:
        print(f"\n❌ Failed to generate token")
        print(f"Status: {response.status_code}")
        print(f"Response: {data}")
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
