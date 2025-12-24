"""
Plotly Dash Web Application for Nifty AI Trading Agent
Production-grade dashboard with real-time charts and analytics
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sys
import os
from datetime import datetime, timedelta
import pytz

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.utils.data_fetcher import get_data_fetcher
from viz.utils.charts import (
    create_win_rate_chart,
    create_confidence_histogram,
    create_signal_distribution_pie,
    create_pnl_timeline,
    create_candlestick_chart
)
from config.settings import TIME_ZONE

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],  # Dark theme
    suppress_callback_exceptions=True
)

app.title = "Nifty Trading Agent | Live Dashboard"

# Initialize data fetcher
fetcher = get_data_fetcher()

# ============================================================================
# LAYOUT
# ============================================================================

# Navbar
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3("📈 Nifty AI Trading Agent", className="text-light mb-0"),
                    html.Small("Live Performance Dashboard", className="text-muted")
                ])
            ], width=8),
            dbc.Col([
                html.Div([
                    html.Span("⚡ Live", className="text-success me-2"),
                    html.Small(id="last-update", className="text-muted")
                ], className="text-end")
            ], width=4)
        ], align="center", className="w-100")
    ], fluid=True),
    color="dark",
    dark=True,
    className="mb-4"
)

# Control Panel
controls = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Time Period:", className="text-light"),
                dcc.Dropdown(
                    id="days-dropdown",
                    options=[
                        {"label": "Last 7 Days", "value": 7},
                        {"label": "Last 14 Days", "value": 14},
                        {"label": "Last 30 Days", "value": 30},
                        {"label": "Last 60 Days", "value": 60},
                        {"label": "Last 90 Days", "value": 90}
                    ],
                    value=30,
                    clearable=False,
                    style={"color": "#000"}
                )
            ], md=3),
            dbc.Col([
                html.Label("Instrument:", className="text-light"),
                dcc.Dropdown(
                    id="instrument-dropdown",
                    options=[
                        {"label": "All", "value": "All"},
                        {"label": "NIFTY 50", "value": "NIFTY 50"},
                        {"label": "BANKNIFTY", "value": "BANKNIFTY"}
                    ],
                    value="All",
                    clearable=False,
                    style={"color": "#000"}
                )
            ], md=3),
            dbc.Col([
                html.Label("Auto Refresh:", className="text-light"),
                dcc.Dropdown(
                    id="refresh-dropdown",
                    options=[
                        {"label": "Off", "value": 0},
                        {"label": "Every 1 min", "value": 60000},
                        {"label": "Every 5 min", "value": 300000},
                        {"label": "Every 10 min", "value": 600000}
                    ],
                    value=0,
                    clearable=False,
                    style={"color": "#000"}
                )
            ], md=3),
            dbc.Col([
                html.Label("\u00A0", className="text-light"),  # Spacer
                dbc.Button(
                    "🔄 Refresh Now",
                    id="refresh-button",
                    color="success",
                    className="w-100"
                )
            ], md=3)
        ])
    ])
], className="mb-4")

# Metrics Cards
metrics_row = html.Div(id="metrics-cards", className="mb-4")

# Charts Row 1: Performance
charts_row1 = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("📊 Win Rate by Signal Type"),
            dbc.CardBody([
                dcc.Graph(id="winrate-chart", config={"displayModeBar": False})
            ])
        ])
    ], md=6),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("🎯 ML Confidence Distribution"),
            dbc.CardBody([
                dcc.Graph(id="confidence-chart", config={"displayModeBar": False})
            ])
        ])
    ], md=6)
], className="mb-4")

# Charts Row 2: Distribution & Timeline
charts_row2 = dbc.Row([
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("📈 Signal Type Distribution"),
            dbc.CardBody([
                dcc.Graph(id="distribution-chart", config={"displayModeBar": False})
            ])
        ])
    ], md=6),
    dbc.Col([
        dbc.Card([
            dbc.CardHeader("💰 Cumulative P&L Timeline"),
            dbc.CardBody([
                dcc.Graph(id="pnl-chart", config={"displayModeBar": False})
            ])
        ])
    ], md=6)
], className="mb-4")

# Recent Trades Table
trades_table = dbc.Card([
    dbc.CardHeader("📋 Recent Trades"),
    dbc.CardBody([
        html.Div(id="trades-table")
    ])
], className="mb-4")

# Auto-refresh interval
interval = dcc.Interval(
    id="interval-component",
    interval=0,  # Controlled by dropdown
    n_intervals=0
)

# Main Layout
app.layout = html.Div([
    navbar,
    dbc.Container([
        controls,
        metrics_row,
        charts_row1,
        charts_row2,
        trades_table,
        interval,
        dcc.Store(id="data-store")  # Store for cached data
    ], fluid=True)
], style={"backgroundColor": "#0d1117"})

# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    [Output("data-store", "data"),
     Output("last-update", "children")],
    [Input("refresh-button", "n_clicks"),
     Input("interval-component", "n_intervals"),
     Input("days-dropdown", "value"),
     Input("instrument-dropdown", "value")]
)
def update_data(n_clicks, n_intervals, days, instrument):
    """Fetch and cache trade data."""
    instr = None if instrument == "All" else instrument
    trades_df = fetcher.fetch_trades(days=days, instrument=instr)
    
    # Convert to dict for storage
    data = {
        "trades": trades_df.to_dict('records') if not trades_df.empty else [],
        "timestamp": datetime.now(pytz.timezone(TIME_ZONE)).strftime("%Y-%m-%d %H:%M:%S IST")
    }
    
    return data, f"Last updated: {data['timestamp']}"

@app.callback(
    Output("metrics-cards", "children"),
    Input("data-store", "data")
)
def update_metrics(data):
    """Update metrics cards."""
    if not data or not data.get("trades"):
        return html.Div("No data available", className="text-center text-muted")
    
    trades_df = pd.DataFrame(data["trades"])
    metrics = fetcher.calculate_performance_metrics(trades_df)
    
    cards = [
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(metrics['total_trades'], className="text-warning"),
                    html.P("Total Trades", className="text-muted mb-0")
                ])
            ], className="text-center")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metrics['win_rate']:.1f}%", 
                           className="text-success" if metrics['win_rate'] >= 60 else "text-danger"),
                    html.P("Win Rate", className="text-muted mb-0")
                ])
            ], className="text-center")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(metrics['wins'], className="text-success"),
                    html.P("Wins", className="text-muted mb-0")
                ])
            ], className="text-center")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(metrics['losses'], className="text-danger"),
                    html.P("Losses", className="text-muted mb-0")
                ])
            ], className="text-center")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(f"{metrics['avg_rr']:.2f}x", className="text-info"),
                    html.P("Avg R:R", className="text-muted mb-0")
                ])
            ], className="text-center")
        ], md=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(metrics['open_trades'], className="text-warning"),
                    html.P("Open", className="text-muted mb-0")
                ])
            ], className="text-center")
        ], md=2)
    ]
    
    return dbc.Row(cards)

@app.callback(
    Output("winrate-chart", "figure"),
    Input("data-store", "data")
)
def update_winrate_chart(data):
    """Update win rate chart."""
    if not data or not data.get("trades"):
        return go.Figure()
    
    trades_df = pd.DataFrame(data["trades"])
    signal_dist = fetcher.get_signal_distribution(trades_df)
    
    return create_win_rate_chart(signal_dist)

@app.callback(
    Output("confidence-chart", "figure"),
    Input("data-store", "data")
)
def update_confidence_chart(data):
    """Update confidence histogram."""
    if not data or not data.get("trades"):
        return go.Figure()
    
    trades_df = pd.DataFrame(data["trades"])
    return create_confidence_histogram(trades_df)

@app.callback(
    Output("distribution-chart", "figure"),
    Input("data-store", "data")
)
def update_distribution_chart(data):
    """Update signal distribution pie chart."""
    if not data or not data.get("trades"):
        return go.Figure()
    
    trades_df = pd.DataFrame(data["trades"])
    signal_dist = fetcher.get_signal_distribution(trades_df)
    
    return create_signal_distribution_pie(signal_dist)

@app.callback(
    Output("pnl-chart", "figure"),
    Input("data-store", "data")
)
def update_pnl_chart(data):
    """Update P&L timeline chart."""
    if not data or not data.get("trades"):
        return go.Figure()
    
    trades_df = pd.DataFrame(data["trades"])
    
    # Convert timestamp if it's string
    if 'timestamp' in trades_df.columns:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    return create_pnl_timeline(trades_df)

@app.callback(
    Output("trades-table", "children"),
    Input("data-store", "data")
)
def update_trades_table(data):
    """Update recent trades table."""
    if not data or not data.get("trades"):
        return html.Div("No trades available", className="text-center text-muted")
    
    trades_df = pd.DataFrame(data["trades"])
    
    # Select columns to display
    display_cols = ['timestamp', 'instrument', 'signal_type', 'entry_price', 
                    'stop_loss', 'take_profit', 'confidence', 'risk_reward', 'status']
    display_cols = [col for col in display_cols if col in trades_df.columns]
    
    # Sort by timestamp (most recent first)
    if 'timestamp' in trades_df.columns:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.sort_values('timestamp', ascending=False)
        trades_df['timestamp'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Limit to 20 most recent
    display_df = trades_df[display_cols].head(20)
    
    # Create table
    table = dbc.Table.from_dataframe(
        display_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True,
        className="table-dark"
    )
    
    return table

@app.callback(
    Output("interval-component", "interval"),
    Input("refresh-dropdown", "value")
)
def update_interval(refresh_value):
    """Update auto-refresh interval."""
    # If 0, disable interval by setting it very high
    return refresh_value if refresh_value > 0 else 1000000000

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8050))
    debug = os.getenv("DASH_DEBUG", "True").lower() == "true"
    
    print(f"\n{'='*60}")
    print(f"🚀 Nifty Trading Agent Dashboard")
    print(f"{'='*60}")
    print(f"📊 Dashboard running at: http://localhost:{port}")
    print(f"🔧 Debug mode: {debug}")
    print(f"{'='*60}\n")
    
    # Updated for Dash 3.x - use app.run() instead of app.run_server()
    app.run(debug=debug, host="0.0.0.0", port=port)
