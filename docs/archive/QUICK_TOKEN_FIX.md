# Quick Fyers Token Refresh - Manual Steps

## The token expired - here's the fastest fix:

### Option 1: Use Fyers Website (2 minutes)

1. **Go to Fyers Dashboard**:
   https://myapi.fyers.in/dashboard

2. **Login** with your credentials

3. **Navigate**: My API Apps → Select your app

4. **Generate Token**: Click "Generate Access Token" button

5. **Copy** the new token (starts with `eyJ...`)

6. **Update Secret Manager**:
```bash
# Replace NEW_TOKEN with the token you copied
echo "NEW_TOKEN_HERE" | gcloud secrets versions add fyers-access-token \
  --data-file=- --project=nifty-trading-agent
```

7. **Done!** Next job run will use new token

---

### Option 2: Use API Script (if you have APP_ID and SECRET_KEY)

The bash script `generate_fyers_token.sh` needs:
- FYERS_APP_ID 
- FYERS_SECRET_KEY

These should be in your `.env` file or Cloud Run environment.

If you have them, run:
```bash
./generate_fyers_token.sh
```

---

## Current Status
- ❌ Fyers API down (token expired)
- ⚠️ System using yfinance fallback
- ❌ No option chain data
- ✅ Basic price data still available

## After Token Updated
System will automatically use new token on next job run (in ~2 minutes)

---

**Fastest**: Use Option 1 (Fyers website) - takes 2 minutes!
