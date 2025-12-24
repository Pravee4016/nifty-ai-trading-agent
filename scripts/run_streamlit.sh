#!/bin/bash
# Launch Streamlit Dashboard

cd "$(dirname "$0")/.."

echo "🚀 Starting Streamlit Dashboard..."
echo "📊 Dashboard will open at http://localhost:8501"
echo ""

./venv/bin/streamlit run viz/streamlit_dashboard.py --server.port 8501
