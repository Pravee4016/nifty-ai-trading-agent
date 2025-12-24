"""
Data Fetcher Utilities for Visualization Dashboards
Fetches trade data and performance metrics from Firestore
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import pytz
from google.cloud import firestore
from google.cloud.firestore import FieldFilter

# Add project root to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import TIME_ZONE

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch trading data from Firestore for visualization."""
    
    def __init__(self):
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.db = None
        
        if self.project_id:
            try:
                self.db = firestore.Client(project=self.project_id)
                logger.info(f"✅ Connected to Firestore: {self.project_id}")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Firestore: {e}")
        else:
            logger.warning("⚠️ GOOGLE_CLOUD_PROJECT not set")
    
    def fetch_trades(self, days: int = 7, instrument: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch trades from Firestore for the last N days.
        
        Args:
            days: Number of days to look back
            instrument: Filter by instrument (NIFTY, BANKNIFTY, etc.)
        
        Returns:
            DataFrame with trade data
        """
        if not self.db:
            logger.warning("No Firestore connection")
            return pd.DataFrame()
        
        try:
            ist = pytz.timezone(TIME_ZONE)
            start_date = (datetime.now(ist) - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Query trades
            trades_ref = self.db.collection("trades")
            query = trades_ref.where(filter=FieldFilter("date", ">=", start_date))
            
            # Filter by instrument if specified
            if instrument:
                query = query.where(filter=FieldFilter("instrument", "==", instrument))
            
            trades = []
            for doc in query.stream():
                trade_data = doc.to_dict()
                trade_data['trade_id'] = doc.id
                trades.append(trade_data)
            
            if not trades:
                logger.info(f"No trades found for last {days} days")
                return pd.DataFrame()
            
            df = pd.DataFrame(trades)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"✅ Fetched {len(df)} trades from last {days} days")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching trades: {e}")
            return pd.DataFrame()
    
    def fetch_daily_stats(self, days: int = 30) -> pd.DataFrame:
        """
        Fetch daily statistics from Firestore.
        
        Args:
            days: Number of days to look back
        
        Returns:
            DataFrame with daily stats
        """
        if not self.db:
            return pd.DataFrame()
        
        try:
            ist = pytz.timezone(TIME_ZONE)
            start_date = (datetime.now(ist) - timedelta(days=days)).strftime("%Y-%m-%d")
            
            stats_ref = self.db.collection("daily_stats")
            query = stats_ref.where(filter=FieldFilter("date", ">=", start_date))
            
            stats = []
            for doc in query.stream():
                stat_data = doc.to_dict()
                stat_data['doc_id'] = doc.id
                stats.append(stat_data)
            
            if not stats:
                return pd.DataFrame()
            
            df = pd.DataFrame(stats)
            logger.info(f"✅ Fetched {len(df)} daily stats")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching daily stats: {e}")
            return pd.DataFrame()
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics from trade data.
        
        Args:
            df: DataFrame with trade data
        
        Returns:
            Dict with performance metrics
        """
        if df.empty:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'avg_rr': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0
            }
        
        # Filter closed trades
        closed_trades = df[df['status'].isin(['WIN', 'LOSS', 'BREAKEVEN'])]
        
        total_trades = len(df)
        wins = len(closed_trades[closed_trades['status'] == 'WIN'])
        losses = len(closed_trades[closed_trades['status'] == 'LOSS'])
        
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
        
        # Calculate average R:R
        avg_rr = df['risk_reward'].mean() if 'risk_reward' in df.columns else 0.0
        
        # Calculate P&L if available
        total_pnl = 0.0
        avg_pnl = 0.0
        if 'outcome' in df.columns:
            trades_with_outcome = df[df['outcome'].notna()]
            if not trades_with_outcome.empty and 'pnl_points' in df.columns:
                total_pnl = df['pnl_points'].sum()
                avg_pnl = df['pnl_points'].mean()
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_rr': avg_rr,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'open_trades': len(df[df['status'] == 'OPEN'])
        }
    
    def get_signal_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get distribution of signal types with win rates.
        
        Args:
            df: DataFrame with trade data
        
        Returns:
            DataFrame with signal type distribution
        """
        if df.empty:
            return pd.DataFrame()
        
        # Group by signal type
        signal_stats = []
        
        for signal_type in df['signal_type'].unique():
            type_df = df[df['signal_type'] == signal_type]
            closed_df = type_df[type_df['status'].isin(['WIN', 'LOSS'])]
            
            wins = len(closed_df[closed_df['status'] == 'WIN'])
            total = len(closed_df)
            win_rate = (wins / total * 100) if total > 0 else 0.0
            
            signal_stats.append({
                'signal_type': signal_type,
                'count': len(type_df),
                'wins': wins,
                'losses': len(closed_df[closed_df['status'] == 'LOSS']),
                'win_rate': win_rate,
                'avg_confidence': type_df['confidence'].mean() if 'confidence' in type_df.columns else 0
            })
        
        return pd.DataFrame(signal_stats).sort_values('count', ascending=False)
    
    def calculate_filter_effectiveness(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze which filters correlate with winning trades.
        
        Args:
            df: DataFrame with trade data (must have 'filters' column)
        
        Returns:
            DataFrame with filter effectiveness
        """
        if df.empty or 'filters' not in df.columns:
            return pd.DataFrame()
        
        # Filter for closed trades only
        closed_trades = df[df['status'].isin(['WIN', 'LOSS'])]
        
        if closed_trades.empty:
            return pd.DataFrame()
        
        filter_stats = {}
        
        for _, trade in closed_trades.iterrows():
            filters = trade.get('filters', {})
            status = trade['status']
            
            for key, value in filters.items():
                feature = f"{key}={value}"
                
                if feature not in filter_stats:
                    filter_stats[feature] = {'wins': 0, 'total': 0}
                
                filter_stats[feature]['total'] += 1
                if status == 'WIN':
                    filter_stats[feature]['wins'] += 1
        
        # Convert to DataFrame
        results = []
        for feature, stats in filter_stats.items():
            if stats['total'] >= 3:  # Minimum sample size
                win_rate = (stats['wins'] / stats['total']) * 100
                results.append({
                    'filter': feature,
                    'win_rate': win_rate,
                    'count': stats['total'],
                    'wins': stats['wins']
                })
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results).sort_values('win_rate', ascending=False)
    
    def get_recent_trades(self, limit: int = 20) -> pd.DataFrame:
        """
        Get most recent trades.
        
        Args:
            limit: Maximum number of trades to return
        
        Returns:
            DataFrame with recent trades
        """
        if not self.db:
            return pd.DataFrame()
        
        try:
            trades_ref = self.db.collection("trades")
            query = trades_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)
            
            trades = []
            for doc in query.stream():
                trade_data = doc.to_dict()
                trade_data['trade_id'] = doc.id
                trades.append(trade_data)
            
            if not trades:
                return pd.DataFrame()
            
            df = pd.DataFrame(trades)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching recent trades: {e}")
            return pd.DataFrame()


# Singleton instance
_fetcher = None

def get_data_fetcher() -> DataFetcher:
    """Get singleton DataFetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = DataFetcher()
    return _fetcher
