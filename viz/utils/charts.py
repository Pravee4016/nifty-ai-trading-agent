"""
Chart Creation Utilities
Reusable Plotly chart components for visualization dashboards
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional


def create_win_rate_chart(signal_dist_df: pd.DataFrame) -> go.Figure:
    """
    Create bar chart showing win rate by signal type.
    
    Args:
        signal_dist_df: DataFrame with columns: signal_type, win_rate, count
    
    Returns:
        Plotly Figure
    """
    if signal_dist_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        return fig
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=signal_dist_df['signal_type'],
        y=signal_dist_df['win_rate'],
        text=signal_dist_df['win_rate'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        marker=dict(
            color=signal_dist_df['win_rate'],
            colorscale='RdYlGn',
            cmin=0,
            cmax=100,
            showscale=False
        ),
        hovertemplate='<b>%{x}</b><br>Win Rate: %{y:.1f}%<br>Count: %{customdata}<extra></extra>',
        customdata=signal_dist_df['count']
    ))
    
    fig.update_layout(
        title="Win Rate by Signal Type",
        xaxis_title="Signal Type",
        yaxis_title="Win Rate (%)",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        height=400
    )
    
    return fig


def create_confidence_histogram(trades_df: pd.DataFrame) -> go.Figure:
    """
    Create histogram of ML confidence scores.
    
    Args:
        trades_df: DataFrame with 'confidence' column
    
    Returns:
        Plotly Figure
    """
    if trades_df.empty or 'confidence' not in trades_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No confidence data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=trades_df['confidence'],
        nbinsx=20,
        marker=dict(color='#1f77b4'),
        hovertemplate='Confidence: %{x}%<br>Count: %{y}<extra></extra>'
    ))
    
    # Add average line
    avg_confidence = trades_df['confidence'].mean()
    fig.add_vline(
        x=avg_confidence,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: {avg_confidence:.1f}%",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="ML Confidence Distribution",
        xaxis_title="Confidence Score (%)",
        yaxis_title="Number of Signals",
        template="plotly_white",
        height=400
    )
    
    return fig


def create_filter_effectiveness_chart(filter_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Create horizontal bar chart showing filter effectiveness.
    
    Args:
        filter_df: DataFrame with columns: filter, win_rate, count
        top_n: Number of top filters to show
    
    Returns:
        Plotly Figure
    """
    if filter_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No filter data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Get top N filters by win rate
    plot_df = filter_df.head(top_n).sort_values('win_rate')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=plot_df['filter'],
        x=plot_df['win_rate'],
        orientation='h',
        text=plot_df['win_rate'].apply(lambda x: f"{x:.1f}%"),
        textposition='outside',
        marker=dict(
            color=plot_df['win_rate'],
            colorscale='RdYlGn',
            cmin=0,
            cmax=100
        ),
        hovertemplate='<b>%{y}</b><br>Win Rate: %{x:.1f}%<br>Count: %{customdata}<extra></extra>',
        customdata=plot_df['count']
    ))
    
    fig.update_layout(
        title=f"Top {min(top_n, len(filter_df))} Filters by Win Rate",
        xaxis_title="Win Rate (%)",
        xaxis=dict(range=[0, 100]),
        yaxis_title="",
        template="plotly_white",
        height=max(400, top_n * 30)
    )
    
    return fig


def create_pnl_timeline(trades_df: pd.DataFrame) -> go.Figure:
    """
    Create line chart of cumulative P&L over time.
    
    Args:
        trades_df: DataFrame with 'timestamp' and 'pnl_points' columns
    
    Returns:
        Plotly Figure
    """
    if trades_df.empty or 'timestamp' not in trades_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No P&L data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Sort by timestamp
    plot_df = trades_df.sort_values('timestamp').copy()
    
    # Calculate cumulative P&L
    if 'pnl_points' in plot_df.columns:
        plot_df['cumulative_pnl'] = plot_df['pnl_points'].cumsum()
    else:
        plot_df['cumulative_pnl'] = 0
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=plot_df['timestamp'],
        y=plot_df['cumulative_pnl'],
        mode='lines+markers',
        line=dict(color='#2ecc71', width=2),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(46, 204, 113, 0.3)',
        hovertemplate='Date: %{x}<br>Cumulative P&L: %{y:.2f} pts<extra></extra>'
    ))
    
    fig.update_layout(
        title="Cumulative P&L Timeline",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L (Points)",
        template="plotly_white",
        height=400
    )
    
    return fig


def create_signal_distribution_pie(signal_dist_df: pd.DataFrame) -> go.Figure:
    """
    Create pie chart of signal type distribution.
    
    Args:
        signal_dist_df: DataFrame with columns: signal_type, count
    
    Returns:
        Plotly Figure
    """
    if signal_dist_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No signal data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=signal_dist_df['signal_type'],
        values=signal_dist_df['count'],
        hole=0.4,
        marker=dict(
            colors=px.colors.qualitative.Set3
        ),
        textposition='auto',
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Signal Type Distribution",
        template="plotly_white",
        height=400
    )
    
    return fig


def create_metrics_cards(metrics: Dict) -> List[Dict]:
    """
    Create metric cards for display.
    
    Args:
        metrics: Dict with performance metrics
    
    Returns:
        List of dicts with metric display info
    """
    cards = [
        {
            'label': 'Total Trades',
            'value': metrics.get('total_trades', 0),
            'icon': '📊',
            'color': '#3498db'
        },
        {
            'label': 'Win Rate',
            'value': f"{metrics.get('win_rate', 0):.1f}%",
            'icon': '🎯',
            'color': '#2ecc71' if metrics.get('win_rate', 0) >= 60 else '#e74c3c'
        },
        {
            'label': 'Wins',
            'value': metrics.get('wins', 0),
            'icon': '✅',
            'color': '#27ae60'
        },
        {
            'label': 'Losses',
            'value': metrics.get('losses', 0),
            'icon': '❌',
            'color': '#e74c3c'
        },
        {
            'label': 'Avg R:R',
            'value': f"{metrics.get('avg_rr', 0):.2f}x",
            'icon': '⚖️',
            'color': '#9b59b6'
        },
        {
            'label': 'Open Trades',
            'value': metrics.get('open_trades', 0),
            'icon': '🔄',
            'color': '#f39c12'
        }
    ]
    
    return cards


def create_candlestick_chart(df: pd.DataFrame, signals: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create candlestick chart with optional signal markers.
    
    Args:
        df: DataFrame with OHLCV data (columns: timestamp/date, open, high, low, close, volume)
        signals: Optional DataFrame with signal data
    
    Returns:
        Plotly Figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No market data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Ensure proper column names (lowercase)
    df_plot = df.copy()
    df_plot.columns = [str(col).lower() for col in df_plot.columns]
    
    # Create subplot with candlestick + volume
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price", "Volume")
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name="Price",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Volume bars
    if 'volume' in df_plot.columns:
        colors = ['#26a69a' if close >= open else '#ef5350' 
                  for close, open in zip(df_plot['close'], df_plot['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df_plot.index,
                y=df_plot['volume'],
                marker_color=colors,
                name="Volume",
                showlegend=False
            ),
            row=2, col=1
        )
    
    # Add signal markers if provided
    if signals is not None and not signals.empty:
        for _, signal in signals.iterrows():
            # Add marker at entry price
            fig.add_trace(
                go.Scatter(
                    x=[signal.get('timestamp', signal.get('entry_time'))],
                    y=[signal['entry_price']],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if 'LONG' in str(signal.get('direction', '')) else 'triangle-down',
                        size=15,
                        color='green' if 'LONG' in str(signal.get('direction', '')) else 'red'
                    ),
                    name=signal.get('signal_type', 'Signal'),
                    showlegend=False,
                    hovertemplate=f"<b>{signal.get('signal_type', 'Signal')}</b><br>" +
                                  f"Entry: {signal['entry_price']:.2f}<br>" +
                                  f"Confidence: {signal.get('confidence', 0):.1f}%<extra></extra>"
                ),
                row=1, col=1
            )
    
    fig.update_layout(
        title="Market Data with Signals",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig
