# Nifty Trading Agent: Deployment Setup

## 🏗️ Architecture Overview

The agent runs as a **Serverless Job** on Google Cloud Platform (GCP).

*   **Compute:** Google Cloud Run (Jobs)
*   **Trigger:** Google Cloud Scheduler
*   **Data:** Google Cloud Firestore (Persistence) & Fyers API (Market Data)
*   **Notifications:** Telegram Bot API
*   **AI:** Groq API

## 📂 Key Deployment Files

### 1. `Dockerfile`
Defines the environment where the agent runs.
*   **Base Image:** `python:3.11-slim` (Lightweight Linux)
*   **Dependencies:** Installs necessary libraries from `requirements.txt`.
*   **Command:** Executes `python main.py` when the container starts.

### 2. `deploy_job.sh`
The master script to deploy code changes.
*   **Builds:** Creates a new Docker image from your code using Google Cloud Build.
*   **Pushes:** Uploads the image to Google Container Registry (GCR).
*   **Updates:** Tells Cloud Run to use this new image for the `trading-agent-job`.
*   **Config:** Injects environment variables from `.env.yaml`.

### 3. `schedule_job.sh`
Configures *when* the agent runs.
*   **Frequency:** Every 5 minutes (`*/5`)
*   **Hours:** 03:45 AM to 10:00 AM UTC (Corresponding to 09:15 AM - 3:30 PM IST).
*   **Days:** Monday to Friday (`1-5`).
*   **Target:** Triggers the Cloud Run Job via HTTP.

## 🚀 How to Deploy

### Standard Update (Code Changes)
Whenever you modify the code (e.g., change logic, fix bugs), run:

```bash
./deploy_job.sh
```

This will build the new code and update the job. The next scheduled run will use the new version.

### Schedule Update (Time Changes)
If you need to change the trading hours or frequency:
1.  Edit `schedule_job.sh`.
2.  Run:
    ```bash
    ./schedule_job.sh
    ```

## ⚙️ Configuration

*   **Environment Variables:** Stored in `.env.yaml`. This file is **sensitive** and contains your API keys (Fyers, Telegram, Groq).
*   **Service Account:** The agent runs under the default compute service account (`499697087516-compute@developer.gserviceaccount.com`), which has valid permissions for Firestore and Logging.

## 📊 Monitoring

*   **Logs:** View logs in the [Google Cloud Console > Cloud Run > Jobs](https://console.cloud.google.com/run/jobs).
*   **Status:** Check the `last_run` status in Cloud Scheduler to ensure it's triggering correctly.
