#!/bin/bash
#
# Deployment script for Nifty AI Trading Agent
# Deploys to Google Cloud Function (Gen 2)
#

set -e

# Configuration
# Use env var if set, otherwise use gcloud config
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
else
    PROJECT_ID="${GOOGLE_CLOUD_PROJECT}"
fi

FUNCTION_NAME="nifty-trading-agent"
REGION="asia-south1"
RUNTIME="python311"
ENTRY_POINT="main"
MEMORY="512MB"
TIMEOUT="540s"

echo "===================================="
echo "Nifty AI Trading Agent - Deployment"
echo "===================================="
echo ""

# Check if project ID is set
if [ -z "$PROJECT_ID" ]; then
    echo "❌ GOOGLE_CLOUD_PROJECT environment variable not set"
    echo "Please set it: export GOOGLE_CLOUD_PROJECT=your-project-id"
    exit 1
fi

echo "📦 Project: $PROJECT_ID"
echo "🌏 Region: $REGION"
echo "⚡ Function: $FUNCTION_NAME"
echo ""

# Run tests first
echo "🧪 Running syntax checks..."
python3 -m py_compile analysis_module/market_state_engine.py
python3 -m py_compile analysis_module/signal_pipeline.py
python3 -m py_compile app/agent.py

echo "✅ Syntax checks passed"
echo ""

# Deploy
echo "🚀 Deploying to Cloud Functions..."
gcloud functions deploy $FUNCTION_NAME \
  --gen2 \
  --region=$REGION \
  --runtime=$RUNTIME \
  --entry-point=$ENTRY_POINT \
  --trigger-http \
  --allow-unauthenticated \
  --memory=$MEMORY \
  --timeout=$TIMEOUT \
  --set-env-vars="USE_ML_FILTERING=False" \
  --source=.

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Monitor logs:"
echo "   gcloud functions logs read $FUNCTION_NAME --region=$REGION --limit=50"
echo ""
echo "🔍 Test function:"
echo "   curl \$(gcloud functions describe $FUNCTION_NAME --region=$REGION --format='value(serviceConfig.uri)')"
echo ""
