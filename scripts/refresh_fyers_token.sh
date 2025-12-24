#!/bin/bash
###############################################################################
# Fyers Token Refresh Script
# Purpose: Quickly refresh Fyers access token when it expires (every 24 hours)
# Usage: ./scripts/refresh_fyers_token.sh
###############################################################################

set -e

echo "🔄 Fyers Token Refresh Script"
echo "=============================="
echo ""

# Configuration
FYERS_CLIENT_ID="DURQKS8D17-100"
FYERS_SECRET_ID="9ZWWXGKO4C"
FYERS_PIN="5679"
REFRESH_TOKEN_FILE=".env"

# Read refresh token from .env file
if [ ! -f "$REFRESH_TOKEN_FILE" ]; then
    echo "❌ Error: .env file not found!"
    exit 1
fi

REFRESH_TOKEN=$(grep "^FYERS_REFRESH_TOKEN=" .env | cut -d'=' -f2)

if [ -z "$REFRESH_TOKEN" ]; then
    echo "❌ Error: FYERS_REFRESH_TOKEN not found in .env file"
    exit 1
fi

echo "✅ Found refresh token in .env"
echo ""

# Generate new access token
echo "🔑 Generating new access token..."
echo ""

RESPONSE=$(python3 << EOF
import hashlib
import requests
import json
import sys

# Generate appIdHash
app_id = "$FYERS_CLIENT_ID"
secret_key = "$FYERS_SECRET_ID"
app_id_hash = hashlib.sha256(f"{app_id}:{secret_key}".encode()).hexdigest()

# API request
url = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
payload = {
    "grant_type": "refresh_token",
    "appIdHash": app_id_hash,
    "refresh_token": "$REFRESH_TOKEN",
    "pin": "$FYERS_PIN"
}

try:
    response = requests.post(url, json=payload)
    result = response.json()
    
    if result.get("code") == 200:
        print(json.dumps(result))
        sys.exit(0)
    else:
        print(json.dumps({"error": result.get("message", "Unknown error")}), file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
EOF
)

if [ $? -ne 0 ]; then
    echo "❌ Failed to generate new token:"
    echo "$RESPONSE"
    exit 1
fi

# Extract new access token
NEW_ACCESS_TOKEN=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('access_token', ''))")

if [ -z "$NEW_ACCESS_TOKEN" ]; then
    echo "❌ Error: Could not extract access token from response"
    echo "$RESPONSE"
    exit 1
fi

echo "✅ New access token generated!"
echo ""

# Backup .env file
BACKUP_FILE=".env.backup_$(date +%Y%m%d_%H%M%S)"
cp .env "$BACKUP_FILE"
echo "📦 Backup created: $BACKUP_FILE"
echo ""

# Update .env file
python3 << EOF
import re

with open('.env', 'r') as f:
    content = f.read()

# Replace access token
content = re.sub(
    r'FYERS_ACCESS_TOKEN=.*',
    f'FYERS_ACCESS_TOKEN=$NEW_ACCESS_TOKEN',
    content
)

with open('.env', 'w') as f:
    f.write(content)

print("✅ .env file updated")
EOF

# Update .env.yaml file if it exists
if [ -f ".env.yaml" ]; then
    cp .env.yaml ".env.yaml.backup_$(date +%Y%m%d_%H%M%S)"
    
    python3 << EOF
with open('.env.yaml', 'r') as f:
    content = f.read()

# Update or add access token
if 'FYERS_ACCESS_TOKEN:' in content:
    import re
    content = re.sub(
        r'FYERS_ACCESS_TOKEN: ".*"',
        f'FYERS_ACCESS_TOKEN: "$NEW_ACCESS_TOKEN"',
        content
    )
else:
    content += f'\nFYERS_ACCESS_TOKEN: "$NEW_ACCESS_TOKEN"\n'

with open('.env.yaml', 'w') as f:
    f.write(content)

print("✅ .env.yaml file updated")
EOF
fi

echo ""
echo "🔄 Updating Google Cloud Secret Manager..."
echo ""

# Update Cloud Secret Manager
echo -n "$NEW_ACCESS_TOKEN" | gcloud secrets versions add fyers-access-token --data-file=- 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Cloud Secret Manager updated successfully"
else
    echo ""
    echo "⚠️  Warning: Failed to update Cloud Secret Manager"
    echo "   You may need to update it manually or check your gcloud authentication"
fi

echo ""
echo "================================"
echo "✅ Token refresh complete!"
echo "================================"
echo ""
echo "📋 Summary:"
echo "   • New access token generated"
echo "   • Local .env files updated"
echo "   • Cloud secrets updated (if gcloud is configured)"
echo ""
echo "⏰ Token valid for: 24 hours"
echo "🔄 Next refresh needed: $(date -v+24H '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "🚀 Next steps:"
echo "   1. For local testing: Token is ready to use"
echo "   2. For production: Deploy or restart Cloud Run job"
echo ""
