#!/bin/bash
# Deploy weekly retrain Cloud Function

set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-nifty-trading-agent}"
REGION="us-central1"
FUNCTION_NAME="nifty-ml-retrain"

echo "========================================="
echo "Deploying ML Retrain Cloud Function"
echo "========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Function: $FUNCTION_NAME"
echo ""

# Deploy function
gcloud functions deploy $FUNCTION_NAME \
    --gen2 \
    --runtime=python311 \
    --region=$REGION \
    --source=./retrain_function \
    --entry-point=retrain_model \
    --trigger-http \
    --allow-unauthenticated \
    --timeout=540s \
    --memory=1GB \
    --set-env-vars="
GOOGLE_CLOUD_PROJECT=$PROJECT_ID,
ML_MODEL_BUCKET=$PROJECT_ID-ml-models,
ML_MIN_TRAINING_SAMPLES=100
" \
    --project=$PROJECT_ID

echo ""
echo "✅ Function deployed successfully!"
echo ""

# Get function URL
FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME \
    --region=$REGION \
    --project=$PROJECT_ID \
    --format='value(serviceConfig.uri)')

echo "Function URL: $FUNCTION_URL"
echo ""
echo "Next step: Set up Cloud Scheduler"
echo "Run: gcloud scheduler jobs create http weekly-ml-retrain \\"
echo "  --schedule='0 21 * * 1' \\"
echo "  --uri='$FUNCTION_URL' \\"
echo "  --http-method=POST \\"
echo "  --time-zone='Asia/Kolkata' \\"
echo "  --location=$REGION \\"
echo "  --project=$PROJECT_ID"
