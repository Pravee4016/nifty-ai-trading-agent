#!/bin/bash
# Deploy Streamlit Dashboard to Google Cloud Run

set -e

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-nifty-trading-agent}"
SERVICE_NAME="nifty-streamlit-dashboard"
REGION="us-central1"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "=========================================="
echo "🚀 Deploying Streamlit Dashboard to Cloud Run"
echo "=========================================="
echo ""
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"
echo "Image: $IMAGE"
echo ""

cd "$(dirname "$0")/.."

# Step 1: Build container with Cloud Build using config file
echo "📦 Building Docker image with Cloud Build..."
gcloud builds submit \
  --config viz/cloudbuild.streamlit.yaml \
  --project $PROJECT_ID \
  .

# Step 2: Deploy to Cloud Run
echo ""
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID \
  --memory 1Gi \
  --cpu 1 \
  --timeout 300 \
  --max-instances 3 \
  --project $PROJECT_ID

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Streamlit Dashboard URL:"
gcloud run services describe $SERVICE_NAME \
  --region $REGION \
  --project $PROJECT_ID \
  --format="value(status.url)"
echo ""
