"""
Streamlit Dashboard for Nifty AI Trading Agent
Real-time performance analytics and trade visualization
"""

import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.utils.data_fetcher import get_data_fetcher
from viz.utils.charts import (
    create_win_rate_chart,
    create_confidence_histogram,
    create_filter_effectiveness_chart,
    create_pnl_timeline,
    create_signal_distribution_pie,
    create_metrics_cards
)

# Page config
st.set_page_config(
    page_title="Nifty Trading Agent Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📈 Nifty AI Trading Agent Dashboard</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

# Date range selector
days_back = st.sidebar.selectbox(
    "Time Period",
    options=[7, 14, 30, 60, 90],
    index=2,
    help="Number of days to look back"
)

# Instrument filter
instrument_filter = st.sidebar.selectbox(
    "Instrument",
    options=["All", "NIFTY 50", "BANKNIFTY"],
    index=0
)

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Initialize data fetcher
@st.cache_resource
def init_fetcher():
    return get_data_fetcher()

fetcher = init_fetcher()

# Fetch data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(days: int, instrument: str):
    """Load trade data from Firestore."""
    instr = None if instrument == "All" else instrument
    trades_df = fetcher.fetch_trades(days=days, instrument=instr)
    return trades_df

with st.spinner("Loading data..."):
    trades_df = load_data(days_back, instrument_filter)

# Check if data loaded successfully
if trades_df.empty:
    st.warning("⚠️ No trade data found. Please check your Firestore connection.")
    st.info("""
    **Troubleshooting:**
    - Ensure `GOOGLE_CLOUD_PROJECT` environment variable is set
    - Verify Firestore service account credentials
    - Check if trades exist in Firestore collection
    """)
    st.stop()

# Calculate metrics
metrics = fetcher.calculate_performance_metrics(trades_df)
signal_dist = fetcher.get_signal_distribution(trades_df)
filter_effectiveness = fetcher.calculate_filter_effectiveness(trades_df)

# ============================================================================
# METRICS CARDS
# ============================================================================

st.markdown("### 📊 Performance Overview")

cols = st.columns(6)

metrics_cards = create_metrics_cards(metrics)

for col, card in zip(cols, metrics_cards):
    with col:
        st.metric(
            label=f"{card['icon']} {card['label']}",
            value=card['value']
        )

st.markdown("---")

# ============================================================================
# CHARTS ROW 1: Win Rate & Confidence
# ============================================================================

st.markdown("### 📈 Signal Performance")

col1, col2 = st.columns(2)

with col1:
    if not signal_dist.empty:
        fig_winrate = create_win_rate_chart(signal_dist)
        st.plotly_chart(fig_winrate, use_container_width=True)
    else:
        st.info("No signal distribution data available")

with col2:
    if not trades_df.empty:
        fig_confidence = create_confidence_histogram(trades_df)
        st.plotly_chart(fig_confidence, use_container_width=True)
    else:
        st.info("No confidence data available")

# ============================================================================
# CHARTS ROW 2: Signal Distribution & P&L Timeline
# ============================================================================

col3, col4 = st.columns(2)

with col3:
    if not signal_dist.empty:
        fig_pie = create_signal_distribution_pie(signal_dist)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No signal data available")

with col4:
    if not trades_df.empty and 'timestamp' in trades_df.columns:
        # Only show P&L timeline if we have pnl_points data
        if 'pnl_points' in trades_df.columns:
            fig_pnl = create_pnl_timeline(trades_df)
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("P&L data not available (trades may not be closed yet)")
    else:
        st.info("No timeline data available")

st.markdown("---")

# ============================================================================
# FILTER EFFECTIVENESS
# ============================================================================

st.markdown("### 🔍 Filter Effectiveness Analysis")

if not filter_effectiveness.empty:
    # Show top N filters selector
    top_n = st.slider("Number of filters to display", min_value=5, max_value=30, value=15, step=5)
    
    fig_filters = create_filter_effectiveness_chart(filter_effectiveness, top_n=top_n)
    st.plotly_chart(fig_filters, use_container_width=True)
    
    # Show data table
    with st.expander("📋 View Filter Data Table"):
        st.dataframe(
            filter_effectiveness.style.background_gradient(cmap='RdYlGn', subset=['win_rate']),
            use_container_width=True
        )
else:
    st.info("No filter effectiveness data available. Ensure trades have 'filters' field in debug_info.")

st.markdown("---")

# ============================================================================
# RECENT TRADES TABLE
# ============================================================================

st.markdown("### 📋 Recent Trades")

# Fetch recent trades
recent_trades = fetcher.get_recent_trades(limit=50)

if not recent_trades.empty:
    # Column selector
    available_cols = recent_trades.columns.tolist()
    default_cols = ['timestamp', 'instrument', 'signal_type', 'entry_price', 'stop_loss', 
                    'take_profit', 'confidence', 'risk_reward', 'status']
    
    # Only show columns that exist
    display_cols = [col for col in default_cols if col in available_cols]
    
    # Format timestamp
    if 'timestamp' in recent_trades.columns:
        recent_trades['timestamp'] = pd.to_datetime(recent_trades['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Color code status
    def highlight_status(row):
        if row['status'] == 'WIN':
            return ['background-color: #d4edda'] * len(row)
        elif row['status'] == 'LOSS':
            return ['background-color: #f8d7da'] * len(row)
        elif row['status'] == 'OPEN':
            return ['background-color: #fff3cd'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = recent_trades[display_cols].style.apply(highlight_status, axis=1)
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Download button
    csv = recent_trades.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"trades_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
else:
    st.info("No recent trades found")

# ============================================================================
# SIGNAL TYPE BREAKDOWN
# ============================================================================

st.markdown("---")
st.markdown("### 📊 Signal Type Breakdown")

if not signal_dist.empty:
    st.dataframe(
        signal_dist.style.background_gradient(cmap='RdYlGn', subset=['win_rate']),
        use_container_width=True
    )
else:
    st.info("No signal distribution data available")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>Nifty AI Trading Agent Dashboard</strong></p>
        <p>Built with Streamlit • Data from Google Cloud Firestore</p>
        <p>Last Updated: {}</p>
    </div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
