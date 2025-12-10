#!/bin/bash
set -e

# Schedule the Cloud Run Job
# Runs every 5 minutes from 09:15 to 15:30 IST (Mon-Fri)

JOB_NAME="trading-agent-job"
SCHEDULER_NAME="trading-agent-scheduler"
REGION="us-central1"
PROJECT_ID="nifty-trading-agent"
SERVICE_ACCOUNT="499697087516-compute@developer.gserviceaccount.com"

# API Endpoint for executing the job
URI="https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/$JOB_NAME:run"

echo "🔄 Updating Cloud Scheduler to target Cloud Run Job..."

# Delete existing if it exists (simplest way to ensure clean switch from Function to Job)
if gcloud scheduler jobs describe $SCHEDULER_NAME --location=$REGION >/dev/null 2>&1; then
    gcloud scheduler jobs delete $SCHEDULER_NAME --location=$REGION --quiet
fi

# Create new scheduler job
gcloud scheduler jobs create http $SCHEDULER_NAME \
    --location=$REGION \
    --schedule="*/5 3-10 * * 1-5" \
    --time-zone="Etc/UTC" \
    --uri="$URI" \
    --http-method=POST \
    --oauth-service-account-email="$SERVICE_ACCOUNT" \
    --headers="Content-Type=application/json" \
    --message-body='{"overrides": {}}' # Empty overrides

echo "✅ Scheduler updated to target Cloud Run Job: $JOB_NAME"
