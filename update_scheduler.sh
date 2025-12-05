#!/bin/bash

# Update Cloud Scheduler to run at appropriate times
# instead of every 5 minutes

SCHEDULER_NAME="trading-agent-scheduler"
LOCATION="us-central1"
FUNCTION_URL="https://us-central1-nifty-trading-agent.cloudfunctions.net/nifty-ai-trading-agent"

echo "🔄 Updating Cloud Scheduler..."

# Update the schedule to run:
# - 15,20,25,30,35 3-10 * * 1-5 (UTC)
# This translates to 09:15, 09:20, 09:25, 09:30, 09:35, then every 5 mins till 15:30 IST
# UTC is 5:30 behind IST, so 09:15 IST = 03:45 UTC

# Better approach: Run every 5 minutes but only during specific windows
# Or keep */5 but rely on the code logic to gate messages

gcloud scheduler jobs update http $SCHEDULER_NAME \
    --location=$LOCATION \
    --schedule="*/5 3-10 * * 1-5" \
    --time-zone="Etc/UTC" \
    --uri="$FUNCTION_URL" \
    --http-method=POST \
    --headers="Content-Type=application/json" \
    --message-body='{}' \
    --oidc-service-account-email="499697087516-compute@developer.gserviceaccount.com"

echo "✅ Scheduler updated successfully"
