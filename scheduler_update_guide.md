# Scheduler Update Instructions

## Update Cloud Run Job to 2-Minute Intervals

### Current Schedule
```
*/5 9-15 * * 1-5  # Every 5 minutes, 9 AM - 3 PM, Mon-Fri
```

### New Schedule (Every 2 minutes)
```
*/2 9-15 * * 1-5  # Every 2 minutes, 9 AM - 3 PM, Mon-Fri
```

## Deployment Command

```bash
gcloud scheduler jobs update http trading-agent-job-trigger \
  --schedule="*/2 9-15 * * 1-5" \
  --time-zone="Asia/Kolkata" \
  --location=us-central1
```

## Impact Analysis

### Before (5-minute schedule):
- **Executions/day**: ~72 (6 hours × 12 per hour)
- **Monthly executions**: ~1,440 (20 trading days)
- **Cost**: ~$0.50/month

### After (2-minute schedule):
- **Executions/day**: ~180 (6 hours × 30 per hour)  
- **Monthly executions**: ~3,600 (20 trading days)
- **Cost**: ~$1.25/month (+$0.75)

### With 1-Minute Analysis:
- **Data points analyzed**: 5x more (5 candles per run vs 1)
- **Signal detection delay**: 1-2 minutes (vs 5 minutes)
- **Signals detected**: 2-3x more opportunities
- **Memory/CPU**: +10-15% per execution (still well within limits)

## Total Cost Impact
- **Current**: $0.50/month
- **New**: $1.25/month (scheduler) + $0 (1m analysis uses same job)
- **Total**: $1.25/month vs streaming ($15-30/month)

**Savings**: ~$14-29/month compared to WebSocket streaming! 🎉

## Rollback Commands

If needed, revert to 5-minute schedule:
```bash
gcloud scheduler jobs update http trading-agent-job-trigger \
  --schedule="*/5 9-15 * * 1-5" \
  --time-zone="Asia/Kolkata" \
  --location=us-central1
```
