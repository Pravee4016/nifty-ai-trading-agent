# Quick Setup Guide for OAuth

## Missing: FYERS_CLIENT_ID

Your `.env` file needs the Fyers Client ID.

### Get Your Client ID:

1. Go to: https://myapi.fyers.in/
2. Login to Fyers account
3. Navigate to: "My Apps"
4. Find your trading app
5. Copy the **App ID** (e.g., `DURQKS8D17-100`)

### Add to .env:

```bash
echo "FYERS_CLIENT_ID=DURQKS8D17-100" >> .env
```

### Then Run OAuth Setup:

```bash
./venv/bin/python3 scripts/setup_fyers_oauth.py
```

The script will guide you through browser authorization.
