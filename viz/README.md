# Visualization Dashboards

Two production-ready visualization dashboards for the Nifty AI Trading Agent.

## Quick Start

### Local Testing

**Streamlit Dashboard**:
```bash
./scripts/run_streamlit.sh
# Opens at http://localhost:8501
```

**Dash Dashboard**:
```bash
./scripts/run_dash.sh
# Opens at http://localhost:8050
```

### Cloud Run Deployment

**Deploy Streamlit**:
```bash
./scripts/deploy_streamlit.sh
```

**Deploy Dash**:
```bash
./scripts/deploy_viz.sh
```

## Features

### Streamlit Dashboard
- 📊 Performance metrics (win rate, R:R, P&L)
- 📈 Interactive charts (win rate by signal type, confidence distribution)
- 🔍 Filter effectiveness analysis
- 📋 Recent trades table with CSV export
- ⚙️ Time period and instrument filters

### Dash Dashboard
- 🌐 Dark theme professional UI
- 📊 Real-time metrics with auto-refresh
- 📈 Interactive Plotly charts
- 🔄 Configurable refresh intervals (1, 5, 10 min)
- 📱 Responsive design

## Documentation

See [docs/VISUALIZATION_GUIDE.md](../docs/VISUALIZATION_GUIDE.md) for complete usage guide.

## Requirements

- Python 3.11+
- Dependencies in `requirements-viz.txt`
- Firestore access with `GOOGLE_CLOUD_PROJECT` env var

## Architecture

```
viz/
├── streamlit_dashboard.py   # Streamlit app
├── dash_app.py              # Dash app
├── utils/
│   ├── data_fetcher.py      # Firestore data access
│   └── charts.py            # Plotly chart components
├── Dockerfile.streamlit     # Streamlit Cloud Run
└── Dockerfile               # Dash Cloud Run
```

## Support

For issues, see [VISUALIZATION_GUIDE.md](../docs/VISUALIZATION_GUIDE.md) troubleshooting section.
