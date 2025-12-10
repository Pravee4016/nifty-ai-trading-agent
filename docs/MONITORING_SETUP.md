# Monitoring Dashboard Setup Guide

## 🎯 Overview

This guide shows you how to set up a monitoring dashboard for the Nifty AI Trading Agent to track adaptive thresholds, win rates, and system health.

---

## 📊 Option 1: Google Cloud Monitoring (Recommended)

### Step 1: Create Custom Metrics

```bash
# 1. Create metrics.yaml
cat > metrics.yaml << 'EOF'
metrics:
  - name: "trading_agent/vix_value"
    type: "GAUGE"
    description: "Current India VIX value"
    
  - name: "trading_agent/rsi_threshold_long"
    type: "GAUGE"
    description: "Adaptive RSI long threshold"
    
  - name: "trading_agent/rsi_threshold_short"
    type: "GAUGE"
    description: "Adaptive RSI short threshold"
    
  - name: "trading_agent/signals_generated"
    type: "COUNTER"
    description: "Total signals generated"
    
  - name: "trading_agent/win_rate"
    type: "GAUGE"
    description: "Current win rate percentage"
EOF
```

### Step 2: Set Up Log-Based Metrics

Go to **Cloud Console → Logging → Logs-based Metrics**:

**VIX Value**:
- Name: `vix_value`
- Filter: `textPayload=~"India VIX: "`
- Metric Type: Last Value
- Field: Extract number after "India VIX: "

**Adaptive RSI (VIX)**:
- Name: `adaptive_rsi_vix`
-Filter: `textPayload=~"Adaptive RSI \(VIX\)"`
- Count occurrences

**Adaptive RSI (ATR)**:
- Name: `adaptive_rsi_atr`
- Filter: `textPayload=~"Adaptive RSI \(ATR\)"`
- Count occurrences

**Fresh Structure Detected**:
- Name: `fresh_structure_count`
- Filter: `textPayload=~"Fresh Structure Detected"`
- Count occurrences

### Step 3: Create Dashboard

1. Go to **Monitoring → Dashboards → Create Dashboard**
2. Name: "Nifty Trading Agent - Performance"
3. Add widgets:

**Widget 1: VIX Over Time**
```
Metric: vix_value
Chart Type: Line
Time Range: Last 24 hours
```

**Widget 2: Adaptive RSI Thresholds**
```
Metrics: 
  - rsi_threshold_long
  - rsi_threshold_short
Chart Type: Line
Time Range: Last 24 hours
```

**Widget 3: Data Source Distribution**
```
Metrics:
  - adaptive_rsi_vix (count)
  - adaptive_rsi_atr (count)
Chart Type: Pie chart
Time Range: Last 7 days
```

**Widget 4: Signal Generation Rate**
```
Metric: signals_generated
Chart Type: Stacked bar
Group by: Pattern type
Time Range: Last 7 days
```

**Widget 5: Win Rate Trend**
```
Metric: win_rate
Chart Type: Line
Time Range: Last 30 days
Threshold: 12% (warning if below)
```

---

## 📊 Option 2: Simple Log Analysis (Quick Start)

### Create Monitoring Script

```bash
cat > monitor_agent.sh << 'EOF'
#!/bin/bash

PROJECT="nifty-trading-agent"
HOURS=${1:-24}

echo "==================================="
echo "Nifty Trading Agent - Last ${HOURS}h"
echo "==================================="

# VIX Statistics
echo -e "\n📊 VIX Statistics:"
gcloud logging read "textPayload=~'India VIX'" \
  --limit=100 \
  --project=$PROJECT \
  --format="value(textPayload)" \
  | grep -oE "[0-9]+\.[0-9]+" \
  | awk '{sum+=$1; count++} END {
    if (count > 0) {
      printf "  Average: %.2f\n", sum/count;
      printf "  Samples: %d\n", count;
    } else {
      print "  No VIX data found";
    }
  }'

# Adaptive Threshold Sources
echo -e "\n🎯 Adaptive Threshold Sources:"
VIX_COUNT=$(gcloud logging read "textPayload=~'Adaptive RSI \(VIX\)'" \
  --limit=100 --project=$PROJECT --format="value(textPayload)" | wc -l)
ATR_COUNT=$(gcloud logging read "textPayload=~'Adaptive RSI \(ATR\)'" \
  --limit=100 --project=$PROJECT --format="value(textPayload)" | wc -l)
  
echo "  VIX-based: $VIX_COUNT"
echo "  ATR-based: $ATR_COUNT"

if [ $VIX_COUNT -gt 0 ] && [ $ATR_COUNT -gt 0 ]; then
  TOTAL=$((VIX_COUNT + ATR_COUNT))
  VIX_PCT=$(echo "scale=1; $VIX_COUNT * 100 / $TOTAL" | bc)
  echo "  VIX Usage: ${VIX_PCT}%"
fi

# Fresh Structure
echo -e "\n🔄 Fresh Structure Detections:"
FRESH_COUNT=$(gcloud logging read "textPayload=~'Fresh Structure Detected'" \
  --limit=100 --project=$PROJECT --format="value(textPayload)" | wc -l)
SUPPRESSED=$(gcloud logging read "textPayload=~'No New Structure'" \
  --limit=100 --project=$PROJECT --format="value(textPayload)" | wc -l)
  
echo "  Allowed: $FRESH_COUNT"
echo "  Suppressed: $SUPPRESSED"

# Option Chain Health
echo -e "\n🛡️ Option Chain Health:"
STALE_COUNT=$(gcloud logging read "textPayload=~'STALE option chain'" \
  --limit=100 --project=$PROJECT --format="value(textPayload)" | wc -l)
DEGRADED=$(gcloud logging read "textPayload=~'DEGRADED MODE'" \
  --limit=100 --project=$PROJECT --format="value(textPayload)" | wc -l)

echo "  Stale data instances: $STALE_COUNT"
echo "  Degraded mode: $DEGRADED"

if [ $STALE_COUNT -eq 0 ] && [ $DEGRADED -eq 0 ]; then
  echo "  ✅ All systems healthy!"
fi

echo -e "\n==================================="
EOF

chmod +x monitor_agent.sh
```

### Usage

```bash
# Monitor last 24 hours (default)
./monitor_agent.sh

# Monitor last 7 days
./monitor_agent.sh 168
```

**Example Output**:
```
===================================
Nifty Trading Agent - Last 24h
===================================

📊 VIX Statistics:
  Average: 15.23
  Samples: 48

🎯 Adaptive Threshold Sources:
  VIX-based: 45
  ATR-based: 3
  VIX Usage: 93.8%

🔄 Fresh Structure Detections:
  Allowed: 12
  Suppressed: 8

🛡️ Option Chain Health:
  Stale data instances: 0
  Degraded mode: 0
  ✅ All systems healthy!
===================================
```

---

## 📊 Option 3: Grafana Dashboard (Advanced)

### Prerequisites

```bash
# Install Grafana
brew install grafana  # macOS
# or
sudo apt-get install grafana  # Linux
```

### Setup

1. **Install Google Cloud Plugin**:
```bash
grafana-cli plugins install grafana-googlecloud-datasource
```

2. **Configure Data Source**:
- Add Google Cloud Monitoring data source
- Project: `nifty-trading-agent`
- Authentication: Use service account key

3. **Import Dashboard Template**:

Create `nifty_agent_dashboard.json`:
```json
{
  "dashboard": {
    "title": "Nifty Trading Agent",
    "panels": [
      {
        "title": "VIX Trend",
        "type": "graph",
        "targets": [
          {
            "metric": "vix_value",
            "aggregation": "mean"
          }
        ]
      },
      {
        "title": "Adaptive RSI Distribution",
        "type": "piechart",
        "targets": [
          {
            "metric": "adaptive_rsi_vix"
          },
          {
            "metric": "adaptive_rsi_atr"
          }
        ]
      },
      {
        "title": "Win Rate",
        "type": "gauge",
        "targets": [
          {
            "metric": "win_rate"
          }
        ],
        "thresholds": [
          { "value": 0, "color": "red" },
          { "value": 10, "color": "yellow" },
          { "value": 15, "color": "green" }
        ]
      }
    ]
  }
}
```

4. **Import**: Dashboards → Import → Upload JSON

---

## 📈 Key Metrics to Track

### Daily Checks

| Metric | Target | Alert If |
|--------|--------|----------|
| VIX Availability | > 90% | < 80% |
| Signals Generated | 5-20/day | < 2 or > 50 |
| Option Chain Health | 100% fresh | > 5% stale |
| System Errors | 0 | > 3 |

### Weekly Analysis

| Metric | Target | Review If |
|--------|--------|-----------|
| Win Rate | 12-20% | < 10% |
| VIX Usage % | > 85% | < 70% |
| Fresh Structure Allows | 20-40% | < 10% or > 60% |
| Adaptive RSI Distribution | Balanced | Stuck at one regime |

---

## 🚨 Alert Configuration

### Set Up Email Alerts

**Google Cloud Monitoring**:

1. Create notification channel:
```bash
gcloud alpha monitoring channels create \
  --display-name="Trading Alert Email" \
  --typeemail \
  --channel-labels=email_address=your@email.com
```

2. Create alerting policies:

**Low Win Rate Alert**:
```yaml
conditions:
  - displayName: "Win Rate Below 10%"
    conditionThreshold:
      filter: metric.type="trading_agent/win_rate"
      comparison: COMPARISON_LT
      thresholdValue: 10
      duration: 86400s  # 24 hours
```

**VIX Fetch Failures**:
```yaml
conditions:
  - displayName: "VIX Unavailable"
    conditionAbsent:
      filter: metric.type="trading_agent/vix_value"
      duration: 3600s  # 1 hour
```

---

## 📊 Quick Dashboard Commands

```bash
# Today's VIX values
gcloud logging read "textPayload=~'India VIX'" \
  --limit=50 --format="table(timestamp, textPayload)"

# Adaptive threshold changes today
gcloud logging read "textPayload=~'Adaptive RSI'" \
  --limit=20 --format="table(timestamp, textPayload)"

# Fresh structure events
gcloud logging read "textPayload=~'Fresh Structure'" \
  --limit=20 --format="table(timestamp, textPayload)"

# System health check
gcloud logging read "severity>=ERROR" \
  --limit=10 --format="table(timestamp, textPayload)"
```

---

## 🎯 Recommended Setup

**For Beginners**:
- Use Option 2 (Simple Log Analysis)
- Run `monitor_agent.sh` daily
- Focus on VIX availability and win rate

**For Advanced Users**:
- Set up Google Cloud Monitoring (Option 1)
- Create custom dashboards
- Configure automated alerts
- Use Grafana for visualization (Option 3)

---

## ✅ Setup Checklist

- [ ] Choose monitoring solution
- [ ] Create log-based metrics
- [ ] Set up dashboard
- [ ] Configure alerts
- [ ] Test with sample queries
- [ ] Schedule daily monitoring
- [ ] Document thresholds
- [ ] Share dashboard with team

---

**Your monitoring dashboard will provide real-time insights into system performance and adaptive behavior!** 📊
