# Fyers OAuth Automation - Complete Setup Guide

## 🎯 Overview

This guide will set up **fully automatic token refresh** for Fyers API using OAuth. Once configured, you'll **never need to manually update tokens again**.

---

## ✨ What You're Getting

✅ **Automatic token refresh** - Tokens refresh before expiry  
✅ **Cloud Secret Manager** - Refresh token stored securely  
✅ **Zero maintenance** - Works indefinitely without intervention  
✅ **Graceful fallback** - Falls back to yfinance if needed  

---

## 📋 Prerequisites

1. **Fyers trading account** with API access
2. **Google Cloud Project**: `nifty-trading-agent`
3. **Environment variables** set:
   - `FYERS_CLIENT_ID`
   - `FYERS_SECRET_ID`

---

## 🚀 Setup Steps

### Step 1: Enable Secret Manager API

```bash
gcloud services enable secretmanager.googleapis.com \
  --project=nifty-trading-agent
```

### Step 2: Grant Secret Manager Permissions

```bash
# Get your Cloud Run service account
gcloud run jobs describe trading-agent-job \
  --region=us-central1 \
  --format='value(serviceAccount)'

# Grant Secret Manager access
gcloud projects add-iam-policy-binding nifty-trading-agent \
  --member=serviceAccount:YOUR_SERVICE_ACCOUNT \
  --role=roles/secretmanager.secretAccessor
```

### Step 3: Run OAuth Setup Script

**Activate virtual environment**:
```bash
source venv/bin/activate
pip install google-cloud-secret-manager
```

**Run setup**:
```bash
python3 scripts/setup_fyers_oauth.py
```

**Follow the prompts**:
1. Browser will open to Fyers authorization page
2. Login and authorize the application
3. Copy the redirect URL
4. Paste into the script
5. Script will:
   - Exchange code for tokens
   - Save refresh token to Secret Manager
   - Update your `.env` file

### Step 4: Deploy with OAuth

```bash
./deploy_job.sh
```

---

## 🔍 How It Works

### Token Lifecycle

```
┌─────────────────────────────────────────────────────┐
│  1. Initial Authorization (One-time Setup)          │
│     - User visits Fyers auth URL                    │
│     - Grants permission                             │
│     - Receives authorization code                   │
│                                                      │
│  2. Token Exchange                                  │
│     - Script exchanges code for:                    │
│       • Access Token (24h lifespan)                 │
│       • Refresh Token (long-lived)                  │
│                                                      │
│  3. Refresh Token Storage                           │
│     - Saved to Cloud Secret Manager                 │
│     - Persists across deployments                   │
│     - Encrypted at rest                             │
│                                                      │
│  4. Automatic Refresh (Every ~23h)                  │
│     ┌─────────────────────────────────────┐        │
│     │  Token about to expire?             │        │
│     │         ↓ YES                       │        │
│     │  Load refresh token from            │        │
│     │  Cloud Secret Manager               │        │
│     │         ↓                            │        │
│     │  Call Fyers refresh endpoint        │        │
│     │         ↓                            │        │
│     │  Get new access token               │        │
│     │         ↓                            │        │
│     │  Continue trading seamlessly        │        │
│     └─────────────────────────────────────┘        │
│                                                      │
│  ∞ Repeats Forever (or until refresh token revoked) │
└─────────────────────────────────────────────────────┘
```

### Code Flow

1. **FyersApp initialization**:
   ```python
   # Automatically tries OAuth first
   if oauth_manager.is_authorized():
       access_token = oauth_manager.get_valid_access_token()
   ```

2. **Token validation**:
   ```python
   # Before each API call
   if token_expiry < now + 10_minutes:
       oauth_manager.refresh_access_token()
   ```

3. **Secret Manager integration**:
   ```python
   # Load refresh token on startup
   refresh_token = secret_manager.get_secret("fyers-refresh-token")
   
   # Save new refresh token if changed
   secret_manager.save_secret("fyers-refresh-token", new_token)
   ```

---

## 🔐 Security

### Secrets Storage

| Secret | Location | Access |
|--------|----------|--------|
| **Refresh Token** | Cloud Secret Manager | Service Account only |
| **Access Token** | Runtime memory | Never persisted |
| **Client ID** | Environment variable | Public (safe) |
| **Secret Key** | Environment variable | Protected |

### Permissions

- **Service Account**: `secretmanager.secretAccessor`
- **Scope**: Project `nifty-trading-agent` only
- **Rotation**: Refresh token rotates on each refresh

---

## 🧪 Testing

### Verify OAuth is Working

```bash
# Check logs for OAuth messages
gcloud run jobs execute trading-agent-job --region=us-central1

# Wait 1 minute, then check logs
gcloud logging read 'textPayload=~"OAuth"' --limit=10
```

**Expected logs**:
```
🔐 Using OAuth manager for automatic token refresh
✅ Using OAuth-managed access token
✅ Access token refreshed successfully
```

### Manually Trigger Refresh

```python
from data_module.fyers_oauth import get_oauth_manager

oauth = get_oauth_manager()
success, message = oauth.refresh_access_token()
print(f"Refresh: {success} - {message}")
print(f"New token: {oauth.access_token[:20]}...")
```

---

## 🛠️ Troubleshooting

### Issue: "No refresh token available"

**Cause**: Setup script not run or failed

**Fix**:
1. Run `python3 scripts/setup_fyers_oauth.py`
2. Complete full authorization flow
3. Verify Secret Manager has `fyers-refresh-token`

### Issue: "Permission denied" on Secret Manager

**Cause**: Service account lacks permissions

**Fix**:
```bash
gcloud projects add-iam-policy-binding nifty-trading-agent \
  --member=serviceAccount:YOUR_SERVICE_ACCOUNT \
  --role=roles/secretmanager.secretAccessor
```

### Issue: Tokens still expiring

**Cause**: OAuth not initialized (missing dependencies)

**Check**:
```bash
# Verify google-cloud-secret-manager is installed
grep "google-cloud-secret-manager" requirements.txt
```

**Fix**:
```bash
# Add to requirements.txt if missing
echo "google-cloud-secret-manager>=2.16.0" >> requirements.txt
./deploy_job.sh
```

---

## 📊 Monitoring

### Check Refresh Token Status

```bash
# List secrets
gcloud secrets list --project=nifty-trading-agent | grep fyers

# View secret metadata (not the value)
gcloud secrets describe fyers-refresh-token --project=nifty-trading-agent
```

### Check Token Refresh Frequency

```bash
# Check how often tokens are refreshed
gcloud logging read 'textPayload=~"token refreshed"' \
  --limit=20 \
  --format="table(timestamp, textPayload)"
```

**Expected**: Every ~23-24 hours

---

## 🔄 Maintenance

### Revoke & Re-authorize

If you need to revoke access and start fresh:

1. **Revoke in Fyers**:
   - Go to https://myapi.fyers.in/
   - Revoke app authorization

2. **Delete secret**:
   ```bash
   gcloud secrets delete fyers-refresh-token --project=nifty-trading-agent
   ```

3. **Re-run setup**:
   ```bash
   python3 scripts/setup_fyers_oauth.py
   ```

### Update Credentials

If Client ID or Secret changes:

1. Update environment variables
2. Delete refresh token
3. Re-run OAuth setup

---

## ✅ Success Checklist

- [ ] Secret Manager API enabled
- [ ] Service account has `secretAccessor` role
- [ ] OAuth setup script run successfully
- [ ] Refresh token in Secret Manager
- [ ] Application deployed
- [ ] Logs show "Using OAuth-managed access token"
- [ ] No "Please provide valid token" errors
- [ ] Tokens auto-refresh every ~24h

---

## 🎉 Result

Once completed:
- ✅ **Zero manual token updates** - Ever!
- ✅ **Tokens refresh automatically** - Every 23-24 hours
- ✅ **Secure storage** - Refresh token in Secret Manager
- ✅ **Graceful fallback** - yfinance if Fyers unavailable
- ✅ **Production-ready** - Enterprise-grade token management

**You're done! Your system will now manage Fyers tokens automatically forever.** 🚀

---

## 📞 Support

**Helper Scripts**:
- `scripts/setup_fyers_oauth.py` - Run OAuth setup
- `scripts/regenerate_fyers_token.sh` - Manual backup option

**Documentation**:
- `docs/FYERS_TOKEN_GUIDE.md` - Token management overview
- `data_module/fyers_oauth.py` - OAuth implementation

**Issues?** Check logs first:
```bash
gcloud run jobs executions list --job=trading-agent-job | head -5
gcloud logging read severity>=ERROR --limit=20
```
