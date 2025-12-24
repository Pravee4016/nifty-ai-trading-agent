#!/bin/bash
set -e

# Deploy updated code to Cloud Run Job
# This updates the container image that the scheduler triggers

PROJECT_ID="nifty-trading-agent"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/trading-agent:latest"
JOB_NAME="trading-agent-job"

echo "🔨 Building and Pushing Docker image via Cloud Build..."
gcloud builds submit --tag $IMAGE_NAME .

echo "🚀 Updating Cloud Run Job..."
# Note: Don't use --env-vars-file as it conflicts with secret references already set
gcloud run jobs update $JOB_NAME \
    --region=$REGION \
    --image=$IMAGE_NAME

echo "🔧 Patching DEPLOYMENT_MODE..."
gcloud run jobs update $JOB_NAME \
    --region=$REGION \
    --update-env-vars="DEPLOYMENT_MODE=GCP"

echo "✅ Cloud Run Job updated successfully!"
echo ""
echo "📊 Job details:"
gcloud run jobs describe $JOB_NAME --region=$REGION --format="yaml(metadata.name,status.latestCreatedExecution)"
