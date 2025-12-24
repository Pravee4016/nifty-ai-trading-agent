# URGENT: Fyers Token Expired - Quick Fix Guide

## Problem
- Token expired at 09:25 AM (market open!)
- System running on yfinance fallback
- No option chain data available

## Quick Fix (5 minutes)

### Step 1: Generate New Token via Fyers Website

1. Go to: https://myapi.fyers.in/dashboard
2. Login with your Fyers credentials
3. Go to "My API Apps"
4. Click on your app (NIFTY-TRADING-AGENT)
5. Click "Generate Access Token"
6. **Copy the new access token** (long string starting with `eyJ...`)

### Step 2: Update Local .env

```bash
# Edit .env file
nano .env

# Find and replace FYERS_ACCESS_TOKEN line with new token:
FYERS_ACCESS_TOKEN=<PASTE_NEW_TOKEN_HERE>

# Save and exit (Ctrl+X, Y, Enter)
```

### Step 3: Update Secret Manager

```bash
cd /Users/praveent/nifty-ai-trading-agent

# Update access token
gcloud secrets versions add fyers-access-token \
  --data-file=- <<< 'PASTE_NEW_TOKEN_HERE'
```

### Step 4: Verify

```bash
# Test locally
python main.py --once

# Look for:
# ✅ Using OAuth-managed access token
# ✅ Fyers Model initialized
```

## Already Done
✅ Refresh token is valid (expires June 6, 2026)
✅ Retrieved from Secret Manager

## Time to Fix
- **Manual via website**: 2-3 minutes
- **Verification**: 1 minute
- **Total**: ~5 minutes

## Why This Happened
- Access tokens expire after ~24 hours
- Auto-refresh needs improvement
- Should monitor token expiry proactively

---

**DO THIS NOW** - Market is open! 🚨
