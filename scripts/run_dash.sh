#!/bin/bash
# Launch Plotly Dash Web App

cd "$(dirname "$0")/.."

echo "🚀 Starting Plotly Dash Dashboard..."
echo "📊 Dashboard will open at http://localhost:8050"
echo ""

export DASH_DEBUG=True

./venv/bin/python viz/dash_app.py
