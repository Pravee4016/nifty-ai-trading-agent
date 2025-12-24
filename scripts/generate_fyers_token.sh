#!/bin/bash

# Fyers Token Generator
# This script generates a new Fyers access token using the refresh token

set -e  # Exit on error

echo "=========================================="
echo "🔑 FYERS TOKEN GENERATOR"
echo "=========================================="

# Load from .env if it exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if we have credentials
if [ -z "$FYERS_APP_ID" ] || [ -z "$FYERS_SECRET_KEY" ]; then
    echo "❌ Missing FYERS_APP_ID or FYERS_SECRET_KEY in .env"
    exit 1
fi

# Get refresh token from Secret Manager if not in env
if [ -z "$FYERS_REFRESH_TOKEN" ]; then
    echo "📥 Fetching refresh token from Secret Manager..."
    FYERS_REFRESH_TOKEN=$(gcloud secrets versions access latest --secret="fyers-refresh-token" --project=nifty-trading-agent)
fi

if [ -z "$FYERS_REFRESH_TOKEN" ]; then
    echo "❌ No refresh token available"
    exit 1
fi

echo "App ID: $FYERS_APP_ID"
echo "Refresh Token: ${FYERS_REFRESH_TOKEN:0:50}..."

# Generate new access token using Fyers API
echo ""
echo "🔄 Generating new access token..."

RESPONSE=$(curl -s -X POST "https://api-t1.fyers.in/api/v3/validate-refresh-token" \
  -H "Content-Type: application/json" \
  -d "{
    \"grant_type\": \"refresh_token\",
    \"appIdHash\": \"$(echo -n "${FYERS_APP_ID}:${FYERS_SECRET_KEY}" | shasum -a 256 | awk '{print $1}')\",
    \"refresh_token\": \"$FYERS_REFRESH_TOKEN\"
  }")

# Parse response
ACCESS_TOKEN=$(echo $RESPONSE | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('access_token', ''))" 2>/dev/null || echo "")

if [ -z "$ACCESS_TOKEN" ]; then
    echo "❌ Failed to generate token"
    echo "Response: $RESPONSE"
    exit 1
fi

echo ""
echo "✅ NEW ACCESS TOKEN GENERATED!"
echo "=========================================="
echo "$ACCESS_TOKEN"
echo "=========================================="
echo "Length: ${#ACCESS_TOKEN}"

# Update .env file
echo ""
echo "📝 Updating .env file..."

if [ -f .env ]; then
    # Backup .env
    cp .env .env.backup
    
    # Update token
    if grep -q "FYERS_ACCESS_TOKEN=" .env; then
        # Replace existing
        sed -i.bak "s|FYERS_ACCESS_TOKEN=.*|FYERS_ACCESS_TOKEN=$ACCESS_TOKEN|" .env
        rm .env.bak
    else
        # Add new
        echo "FYERS_ACCESS_TOKEN=$ACCESS_TOKEN" >> .env
    fi
    
    echo "✅ .env updated (backup saved as .env.backup)"
else
    echo "⚠️ No .env file found, creating one..."
    echo "FYERS_ACCESS_TOKEN=$ACCESS_TOKEN" > .env
fi

# Update Secret Manager
echo ""
echo "📤 Updating Google Secret Manager..."

echo "$ACCESS_TOKEN" | gcloud secrets versions add fyers-access-token \
  --data-file=- \
  --project=nifty-trading-agent

if [ $? -eq 0 ]; then
    echo "✅ Secret Manager updated"
else
    echo "❌ Failed to update Secret Manager"
    exit 1
fi

# Test the token
echo ""
echo "🧪 Testing new token..."

TEST_RESPONSE=$(curl -s "https://api-t1.fyers.in/api/v3/profile" \
  -H "Authorization: ${FYERS_APP_ID}:${ACCESS_TOKEN}")

if echo "$TEST_RESPONSE" | grep -q '"s":"ok"'; then
    echo "✅ Token is working!"
    NAME=$(echo $TEST_RESPONSE | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('data', {}).get('name', 'N/A'))" 2>/dev/null || echo "N/A")
    echo "Account: $NAME"
else
    echo "⚠️ Token test failed"
    echo "Response: $TEST_RESPONSE"
fi

echo ""
echo "=========================================="
echo "✅ DONE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Token is already updated in Secret Manager ✅"
echo "2. Restart Cloud Run job to use new token"
echo ""
echo "Restart command:"
echo "  gcloud run jobs execute trading-agent-job --region=us-central1"
