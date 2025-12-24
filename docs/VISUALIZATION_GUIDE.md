# Visualization Dashboards - User Guide

## 📊 Overview

The Nifty AI Trading Agent now includes two powerful visualization dashboards:

1. **Streamlit Dashboard** - Quick internal analytics (runs locally)
2. **Plotly Dash Web App** - Production-grade dashboard with auto-refresh

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+ with virtual environment activated
- Google Cloud Firestore access (trades stored in `trades` collection)
- Environment variable `GOOGLE_CLOUD_PROJECT` set

### Installation

Dependencies are already installed if you ran:
```bash
pip install -r requirements-viz.txt
```

---

## 1️⃣ Streamlit Dashboard (Quick Win)

### Launch Command

```bash
./scripts/run_streamlit.sh
```

Or manually:
```bash
streamlit run viz/streamlit_dashboard.py --server.port 8501
```

### Access

Open in browser: **http://localhost:8501**

### Features

- **Performance Overview**: Total trades, win rate, wins/losses, avg R:R
- **Win Rate by Signal Type**: Bar chart showing performance per pattern
- **ML Confidence Distribution**: Histogram of signal confidence scores
- **Signal Distribution**: Pie chart of signal types
- **P&L Timeline**: Cumulative profit/loss over time
- **Filter Effectiveness**: Which filters correlate with winning trades
- **Recent Trades Table**: Last 50 trades with status highlighting

### Filters

- **Time Period**: 7, 14, 30, 60, or 90 days
- **Instrument**: All, NIFTY 50, or BANKNIFTY

### Data Refresh

- Click **"🔄 Refresh Data"** button in sidebar
- Data is cached for 5 minutes

---

## 2️⃣ Plotly Dash Web App (Production)

### Launch Command (Local)

```bash
./scripts/run_dash.sh
```

Or manually:
```bash
python viz/dash_app.py
```

### Access

Open in browser: **http://localhost:8050**

### Features

- **Dark Theme**: Professional trading platform aesthetic
- **Real-time Metrics**: Auto-updating KPI cards
- **Interactive Charts**: Zoom, pan, hover tooltips
- **Auto-Refresh**: Configurable refresh intervals (1, 5, or 10 minutes)
- **Responsive Design**: Works on desktop and mobile

### Control Panel

| Control | Options | Description |
|---------|---------|-------------|
| **Time Period** | 7, 14, 30, 60, 90 days | Historical data range |
| **Instrument** | All, NIFTY 50, BANKNIFTY | Filter by index |
| **Auto Refresh** | Off, 1min, 5min, 10min | Auto-update interval |
| **Refresh Now** | Button | Manual refresh |

### Charts

1. **Win Rate by Signal Type** - Bar chart with color gradient
2. **ML Confidence Distribution** - Histogram with average line
3. **Signal Type Distribution** - Donut chart
4. **Cumulative P&L Timeline** - Line chart with area fill
5. **Recent Trades Table** - Last 20 trades

---

## ☁️ Cloud Deployment (Dash Only)

### Deploy to Cloud Run

```bash
./scripts/deploy_viz.sh
```

This will:
- Build Docker container from `viz/Dockerfile`
- Deploy to Cloud Run service `nifty-viz-dashboard`
- Make it publicly accessible
- Return the dashboard URL

### Production URL

After deployment, access at:
```
https://nifty-viz-dashboard-[hash]-uc.a.run.app
```

### Environment Variables (Cloud Run)

Set in deployment:
- `GOOGLE_CLOUD_PROJECT`: Your GCP project ID
- `PORT`: 8080 (default for Cloud Run)
- `DASH_DEBUG`: False (production mode)

---

## 🔧 Troubleshooting

### No Data Showing

**Problem**: Dashboard loads but shows "No data available"

**Solutions**:
1. Check Firestore connection:
   ```bash
   echo $GOOGLE_CLOUD_PROJECT
   # Should print: nifty-trading-agent
   ```

2. Verify service account credentials:
   ```bash
   ls trading-agent-key.json
   # File should exist
   ```

3. Check if trades exist in Firestore:
   ```python
   from google.cloud import firestore
   db = firestore.Client(project="nifty-trading-agent")
   trades = list(db.collection("trades").limit(5).stream())
   print(f"Trades found: {len(trades)}")
   ```

### Streamlit Connection Error

**Problem**: `OSError: [Errno 48] Address already in use`

**Solution**: Port 8501 is occupied. Kill the process or use a different port:
```bash
streamlit run viz/streamlit_dashboard.py --server.port 8502
```

### Dash App Not Starting

**Problem**: Import errors or dependency issues

**Solution**: Reinstall visualization dependencies:
```bash
pip install -r requirements-viz.txt --force-reinstall
```

### Charts Not Rendering

**Problem**: Plotly charts show blank

**Solution**:
1. Clear browser cache
2. Check JavaScript console for errors (F12)
3. Ensure `plotly>=5.18.0` is installed

---

## 📁 File Structure

```
nifty-ai-trading-agent/
├── viz/
│   ├── streamlit_dashboard.py    # Streamlit app
│   ├── dash_app.py                # Dash app
│   ├── Dockerfile                 # Cloud Run container
│   └── utils/
│       ├── data_fetcher.py        # Firestore data access
│       └── charts.py              # Plotly chart components
├── scripts/
│   ├── run_streamlit.sh           # Launch Streamlit
│   ├── run_dash.sh                # Launch Dash
│   └── deploy_viz.sh              # Deploy to Cloud Run
└── requirements-viz.txt           # Viz dependencies
```

---

## 🎨 Customization

### Change Streamlit Theme

Edit `viz/streamlit_dashboard.py` CSS section:
```python
st.markdown("""
    <style>
    .main-header {
        color: #YOUR_COLOR;  # Change header color
    }
    </style>
""", unsafe_allow_html=True)
```

### Change Dash Theme

In `viz/dash_app.py`, replace `dbc.themes.CYBORG` with:
- `dbc.themes.DARKLY`
- `dbc.themes.SLATE`
- `dbc.themes.SUPERHERO`

### Add New Charts

1. Create chart function in `viz/utils/charts.py`
2. Add graph component to layout
3. Add callback to update the chart

Example:
```python
# In charts.py
def create_my_chart(data):
    fig = go.Figure()
    # Build your chart
    return fig

# In streamlit_dashboard.py or dash_app.py
st.plotly_chart(create_my_chart(data))
```

---

## 📊 Data Sources

Both dashboards fetch data from:

- **Firestore Collection**: `trades`
- **Required Fields**:
  - `timestamp`: Trade date/time
  - `instrument`: NIFTY 50 / BANKNIFTY
  - `signal_type`: Pattern type
  - `entry_price`, `stop_loss`, `take_profit`
  - `confidence`: ML confidence score
  - `status`: OPEN, WIN, LOSS, BREAKEVEN
  - `risk_reward`: R:R ratio
  - `filters`: Debug info (for filter effectiveness)

---

## 🔒 Security Notes

### Streamlit (Local Only)

- Runs on localhost by default
- No authentication required
- Safe for internal use

### Dash (Cloud Run)

- Currently deployed with `--allow-unauthenticated`
- **Recommendation**: Add authentication for production
  - Use Cloud IAP (Identity-Aware Proxy)
  - Add login page with Dash Auth
  - Restrict by IP allowlist

### Environment Variables

Never commit:
- `GOOGLE_CLOUD_PROJECT` to public repos
- Service account keys
- `.env` files

---

## 📈 Performance Tips

### Streamlit

- Data is cached for 5 minutes (`@st.cache_data(ttl=300)`)
- Use "Refresh Data" button to force reload
- Limit date range for faster loading

### Dash

- Auto-refresh disabled by default (set to "Off")
- Enable only during market hours
- Use 5-10 minute intervals to reduce Firestore reads

---

## 🆘 Support

For issues:
1. Check Cloud Functions logs for data issues
2. Review Firestore console for missing trades
3. Test data fetcher independently:
   ```python
   from viz.utils.data_fetcher import get_data_fetcher
   fetcher = get_data_fetcher()
   df = fetcher.fetch_trades(days=7)
   print(df.head())
   ```

---

**Last Updated**: 2025-12-21  
**Version**: 1.0.0
