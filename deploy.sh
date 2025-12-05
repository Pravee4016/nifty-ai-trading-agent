#!/bin/bash

# Deployment Script for Nifty AI Trading Agent
# Deploys to Google Cloud Functions (Gen 2)

FUNCTION_NAME="nifty-ai-trading-agent"
REGION="us-central1"
ENTRY_POINT="cloud_function_handler"
RUNTIME="python310"
MEMORY="512MB"
TIMEOUT="540s"

echo "🚀 Deploying $FUNCTION_NAME to $REGION..."

gcloud functions deploy $FUNCTION_NAME \
    --gen2 \
    --region=$REGION \
    --runtime=$RUNTIME \
    --entry-point=$ENTRY_POINT \
    --source=. \
    --trigger-http \
    --allow-unauthenticated \
    --memory=$MEMORY \
    --timeout=$TIMEOUT \
    --env-vars-file=.env.yaml

echo "✅ Deployment command finished."
