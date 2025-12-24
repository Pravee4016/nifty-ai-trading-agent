# Google Cloud Monitoring Dashboard - Setup Guide

## 🎯 Complete Step-by-Step Instructions

**Time Required**: 20-30 minutes  
**Prerequisites**: 
- Google Cloud Console access
- Project: `nifty-trading-agent`
- Trading agent already deployed

---

## Step 1: Create Log-Based Metrics

### 1.1 Navigate to Logs Explorer

1. Open [Google Cloud Console](https://console.cloud.google.com)
2. Select project: **nifty-trading-agent**
3. Go to: **Logging** → **Logs Explorer**

### 1.2 Create VIX Value Metric

**Query**:
```
resource.type="cloud_run_job"
resource.labels.job_name="trading-agent-job"
textPayload=~"India VIX: [0-9]+\.[0-9]+"
```

**Steps**:
1. Paste query in Logs Explorer
2. Click **"Create metric"** (top right)
3. **Metric type**: Distribution
4. **Metric name**: `vix_value`
5. **Field name**: `textPayload`
6. **Regular expression**: `India VIX: ([0-9]+\.[0-9]+)`
7. **Extraction**: `$1`
8. **Type**: Double
9. Click **Create Metric**

### 1.3 Create Adaptive RSI VIX Count

**Query**:
```
resource.type="cloud_run_job"
resource.labels.job_name="trading-agent-job"
textPayload=~"Adaptive RSI \(VIX\)"
```

**Steps**:
1. Paste query
2. Click **"Create metric"**
3. **Metric type**: Counter
4. **Metric name**: `adaptive_rsi_vix_count`
5. **Description**: "Count of VIX-based threshold calculations"
6. Click **Create Metric**

### 1.4 Create Adaptive RSI ATR Count  

**Query**:
```
resource.type="cloud_run_job"
resource.labels.job_name="trading-agent-job"
textPayload=~"Adaptive RSI \(ATR\)"
```

**Steps**:
1. Paste query
2. **Metric type**: Counter
3. **Metric name**: `adaptive_rsi_atr_count`
4. Click **Create Metric**

### 1.5 Create Fresh Structure Count

**Query**:
```
resource.type="cloud_run_job"
resource.labels.job_name="trading-agent-job"
textPayload=~"Fresh Structure Detected"
```

**Steps**:
1. Paste query
2. **Metric type**: Counter
3. **Metric name**: `fresh_structure_count`
4. Click **Create Metric**

### 1.6 Create Error Count

**Query**:
```
resource.type="cloud_run_job"
resource.labels.job_name="trading-agent-job"
severity>=ERROR
```

**Steps**:
1. Paste query
2. **Metric type**: Counter
3. **Metric name**: `error_count`
4. Click **Create Metric**

---

## Step 2: Create Monitoring Dashboard

### 2.1 Navigate to Dashboards

1. Go to: **Monitoring** → **Dashboards**
2. Click **"+ Create Dashboard"**
3. Name: **"Nifty Trading Agent - Performance"**

### 2.2 Add Chart: VIX Trend

1. Click **"+ Add Chart"**
2. **Chart type**: Line chart
3. **Title**: "India VIX - Last 24 Hours"

**Configuration**:
- **Resource type**: Cloud Run Job
- **Metric**: `logging.googleapis.com/user/vix_value`
- **Filter**: `job_name = "trading-agent-job"`
- **Aggregation**: Mean
- **Alignment period**: 5 minutes
- **Time Range**: 24 hours

4. Click **Save**

### 2.3 Add Chart: VIX vs ATR Usage

1. Click **"+ Add Chart"**
2. **Chart type**: Pie chart
3. **Title**: "Adaptive Threshold Source Distribution"

**Add Metrics**:

**Metric 1 (VIX)**:
- **Metric**: `logging.googleapis.com/user/adaptive_rsi_vix_count`
- **Label**: "VIX-based"
- **Aggregation**: Sum
- **Time Range**: 7 days

**Metric 2 (ATR)**:
- **Metric**: `logging.googleapis.com/user/adaptive_rsi_atr_count`
- **Label**: "ATR-based"
- **Aggregation**: Sum
- **Time Range**: 7 days

4. Click **Save**

### 2.4 Add Chart: Fresh Structure Activity

1. Click **"+ Add Chart"**
2. **Chart type**: Stacked bar chart
3. **Title**: "Fresh Structure Detections"

**Configuration**:
- **Metric**: `logging.googleapis.com/user/fresh_structure_count`
- **Aggregation**: Rate (per minute)
- **Alignment**: 1 hour
- **Time Range**: 7 days

4. Click **Save**

### 2.5 Add Chart: System Health (Errors)

1. Click **"+ Add Chart"**
2. **Chart type**: Scorecard
3. **Title**: "Errors (Last 24h)"

**Configuration**:
- **Metric**: `logging.googleapis.com/user/error_count`
- **Aggregation**: Sum
- **Time Range**: 24 hours
- **Thresholds**: 
  - Green: 0
  - Yellow: > 0
  - Red: > 5

4. Click **Save**

### 2.6 Add Chart: Log Volume

1. Click **"+ Add Chart"**
2. **Chart type**: Line chart
3. **Title**: "Analysis Cycles (Log Activity)"

**Configuration**:
- **Resource**: Cloud Run Job
- **Metric**: `run.googleapis.com/request_count`
- **Filter**: `job_name = "trading-agent-job"`
- **Aggregation**: Rate
- **Time Range**: 24 hours

4. Click **Save**

---

## Step 3: Set Up Alerting

### 3.1 Create Notification Channel

1. Go to: **Monitoring** → **Alerting** → **Notification Channels**
2. Click **"+ New Channel"**
3. **Type**: Email
4. **Email**: `your-email@example.com`
5. **Display name**: "Trading Alert Email"
6. Click **Save**

### 3.2 Create Alert: High Error Rate

1. Go to: **Monitoring** → **Alerting** → **Create Policy**
2. **Name**: "Trading Agent - High Error Rate"

**Condition**:
- **Target**: Cloud Run Job
- **Metric**: `logging.googleapis.com/user/error_count`
- **Filter**: `job_name = "trading-agent-job"`
- **Threshold**: > 5 errors
- **Duration**: 15 minutes

**Notification**:
- Channel: Your email
- **Documentation**:
  ```
  The trading agent has logged more than 5 errors in the last 15 minutes.
  
  Check logs: https://console.cloud.google.com/logs
  ```

3. Click **Save**

### 3.3 Create Alert: VIX Data Missing

1. Click **"+ Create Policy"**
2. **Name**: "Trading Agent - VIX Data Unavailable"

**Condition**:
- **Type**: Metric absence
- **Metric**: `logging.googleapis.com/user/vix_value`
- **Duration**: 1 hour

**Notification**:
- Channel: Your email
- **Documentation**:
  ```
  VIX data has not been logged for the past hour.
  The system should fall back to ATR percentile automatically.
  
  Check if yfinance and Fyers API are both failing.
  ```

3. Click **Save**

---

## Step 4: Test Your Dashboard

### 4.1 Verify Metrics Are Collecting

1. Go to **Metrics Explorer**
2. **Resource**: Cloud Run Job
3. Search for: `vix_value`
4. You should see data points if the agent has run recently

### 4.2 Trigger a Test Run

```bash
# Manually execute the job
gcloud run jobs execute trading-agent-job \
  --region us-central1 \
  --project nifty-trading-agent
```

### 4.3 Wait 5-10 Minutes

Allow time for:
- Job to complete
- Logs to be ingested
- Metrics to be calculated

### 4.4 Check Dashboard

1. Go to your dashboard
2. Refresh the page
3. Verify charts are populating

---

## Step 5: Customize & Enhance

### 5.1 Adjust Time Ranges

Each chart can have custom time ranges:
- **VIX**: Last 24 hours (good for short-term trends)
- **Source Distribution**: Last 7 days (see overall pattern)
- **Errors**: Last 7 days (spot patterns)

### 5.2 Add Threshold Lines

For VIX chart:
1. Edit chart
2. Click **"+ Add threshold"**
3. **Value**: 12 (Choppy threshold)
4. **Label**: "Choppy/Normal boundary"
5. **Value**: 18 (Volatile threshold)
6. **Label**: "Normal/Volatile boundary"

### 5.3 Create Custom Metrics

**Example: Win Rate (Manual Entry)**

Since win rate is calculated from Telegram alerts, you can:
1. Create a custom metric manually
2. Use Cloud Scheduler to update it
3. Or track in a spreadsheet and review weekly

---

## Step 6: Access & Share Dashboard

### 6.1 Get Dashboard URL

1. Open your dashboard
2. Copy the URL
3. Bookmarkit for quick access

**Example**:
```
https://console.cloud.google.com/monitoring/dashboards/custom/12345678
```

### 6.2 Share with Team

1. Click **"Share"** (top right)
2. Add team members' emails
3. Set permissions (Viewer/Editor)

---

## 📊 Expected Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│        Nifty Trading Agent - Performance                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────┐  ┌──────────────────────┐   │
│  │   VIX Trend         │  │ Source Distribution   │   │
│  │   (Last 24h)         │  │   VIX: 93%           │   │
│  │                      │  │   ATR: 7%            │   │
│  │   ~~~~/\~~~~         │  │                      │   │
│  └──────────────────────┘  └──────────────────────┘   │
│                                                         │
│  ┌──────────────────────┐  ┌──────────────────────┐   │
│  │ Fresh Structure      │  │   Errors (24h)       │   │
│  │   (Last 7 days)      │  │                      │   │
│  │   ▂▃▄▅▃▂            │  │       0              │   │
│  │                      │  │     ✅ OK            │   │
│  └──────────────────────┘  └──────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐ │
│  │        Analysis Cycles (Log Activity)             │ │
│  │        (Last 24h)                                 │ │
│  │        ▃▄█▅▆▃▂▁                                  │ │
│  └──────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Troubleshooting

### Issue: No Data in Charts

**Cause**: Metrics not collecting or job not running

**Fix**:
1. Check if job has run recently:
   ```bash
   gcloud run jobs executions list \
     --job trading-agent-job \
     --region us-central1
   ```

2. Verify log-based metrics exist:
   - Go to **Logging** → **Logs-based metrics**
   - Check if your metrics are listed

3. Re-run the job manually (see Step 4.2)

### Issue: Metrics Showing Wrong Data

**Cause**: Regular expression not matching logs

**Fix**:
1. Test your query in Logs Explorer
2. Verify log format matches regex
3. Update metric extraction pattern

### Issue: Alerts Not Firing

**Cause**: Notification channel not configured

**Fix**:
1. Verify channel: **Alerting** → **Notification Channels**
2. Check spam folder for test emails
3. Verify alert conditions are met

---

## ✅ Setup Checklist

- [ ] Created 5+ log-based metrics
- [ ] Built dashboard with 5+ charts
- [ ] Set up notification channel (email)
- [ ] Created 2+ alert policies
- [ ] Tested dashboard with manual job run
- [ ] Verified charts are populating
- [ ] Bookmarked dashboard URL
- [ ] Shared with team (if applicable)

---

## 📈 Next: Monitor for 7 Days

1. **Check dashboard daily**
2. **Track VIX availability**: Should be > 90%
3. **Monitor error scorecard**: Should stay green (0)
4. **Review source distribution**: VIX should dominate
5. **Note any alerts**: Investigate immediately

---

**Your Google Cloud Monitoring dashboard is now live!** 🎉

Access it anytime to track your trading agent's performance and health.
