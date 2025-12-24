import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from analysis_module.technical import TechnicalAnalyzer

def reproduce_orb_bug():
    print("Running ORB Bug Reproduction...")
    
    # Create mock data for 2 days
    # Day 1: 2024-12-11 (Stale) - High prices
    # Day 2: 2024-12-12 (Today)  - Current prices
    
    dates_day1 = pd.date_range("2024-12-11 09:15:00", "2024-12-11 15:30:00", freq="15min")
    dates_day2 = pd.date_range("2024-12-12 09:15:00", "2024-12-12 15:30:00", freq="15min")
    
    df_day1 = pd.DataFrame({
        "open": np.linspace(26000, 26100, len(dates_day1)),
        "high": np.linspace(26010, 26110, len(dates_day1)),
        "low": np.linspace(25990, 26090, len(dates_day1)),
        "close": np.linspace(26005, 26105, len(dates_day1)),
        "volume": 1000
    }, index=dates_day1)
    
    df_day2 = pd.DataFrame({
        "open": np.linspace(25800, 25900, len(dates_day2)),
        "high": np.linspace(25810, 25910, len(dates_day2)),
        "low": np.linspace(25790, 25890, len(dates_day2)),
        "close": np.linspace(25805, 25905, len(dates_day2)),
        "volume": 1000
    }, index=dates_day2)
    
    df = pd.concat([df_day1, df_day2])
    
    analyzer = TechnicalAnalyzer("NIFTY")
    orb = analyzer.get_opening_range(df, duration_mins=15)
    
    print("\nResults:")
    print(f"Latest Date in Data: {df.index[-1].date()}")
    print(f"ORB High identified: {orb['high']}")
    
    if orb['high'] > 26000:
        print("\n❌ BUG CONFIRMED: Opening Range picked from Day 1 (26000s) instead of Today (25800s)")
    else:
        print("\n✅ BUG NOT FOUND: Opening Range picked from Today")

if __name__ == "__main__":
    reproduce_orb_bug()
