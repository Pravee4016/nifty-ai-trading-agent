#!/bin/bash

# Deploy updated code to Cloud Run Job
# This updates the container image that the scheduler triggers

PROJECT_ID="nifty-trading-agent"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/trading-agent:latest"
JOB_NAME="trading-agent-job"

echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME .

echo "📤 Pushing image to Google Container Registry..."
docker push $IMAGE_NAME

echo "🚀 Updating Cloud Run Job..."
gcloud run jobs update $JOB_NAME \
    --region=$REGION \
    --image=$IMAGE_NAME

echo "✅ Cloud Run Job updated successfully!"
