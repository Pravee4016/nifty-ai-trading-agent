"""
Analyze Filter Effectiveness
Script to analyze which filters are contributing to win rate and which are not.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import firestore

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import TIME_ZONE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_filters(days: int = 30):
    """Analyze filter effectiveness from Firestore trades."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        logger.error("GOOGLE_CLOUD_PROJECT not set")
        return

    db = firestore.Client(project=project_id)
    
    # Fetch trades
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    trades_ref = db.collection("trades")
    query = trades_ref.where("date", ">=", start_date).stream()
    
    trades = []
    for doc in query:
        trades.append(doc.to_dict())
        
    if not trades:
        logger.info("No trades found for analysis.")
        return

    df = pd.DataFrame(trades)
    
    # Filter for closed trades
    closed_trades = df[df["status"].isin(["WIN", "LOSS"])]
    
    if closed_trades.empty:
        logger.info("No closed trades found.")
        return

    logger.info(f"Analyzing {len(closed_trades)} closed trades...")
    
    # 1. Overall Win Rate
    win_rate = (len(closed_trades[closed_trades["status"] == "WIN"]) / len(closed_trades)) * 100
    logger.info(f"Overall Win Rate: {win_rate:.2f}%")
    
    # 2. Analyze Filters
    # Extract filters from 'filters' column (dict)
    # We'll look for common keys like 'is_consolidating', 'volume_confirmed', 'trend_dir'
    
    filter_stats = {}
    
    # Flatten filters
    for _, trade in closed_trades.iterrows():
        filters = trade.get("filters", {})
        status = trade["status"]
        
        for key, value in filters.items():
            # Create a feature key (e.g., "consolidating=True")
            feature = f"{key}={value}"
            
            if feature not in filter_stats:
                filter_stats[feature] = {"wins": 0, "total": 0}
            
            filter_stats[feature]["total"] += 1
            if status == "WIN":
                filter_stats[feature]["wins"] += 1
                
    # Print Stats
    logger.info("\nFilter Effectiveness:")
    logger.info(f"{'Filter':<40} | {'Win Rate':<10} | {'Count':<5}")
    logger.info("-" * 65)
    
    sorted_stats = sorted(
        filter_stats.items(), 
        key=lambda x: (x[1]["wins"] / x[1]["total"]), 
        reverse=True
    )
    
    for feature, stats in sorted_stats:
        if stats["total"] < 3:  # Skip low sample size
            continue
            
        wr = (stats["wins"] / stats["total"]) * 100
        logger.info(f"{feature:<40} | {wr:6.2f}%   | {stats['total']:<5}")

if __name__ == "__main__":
    analyze_filters()
