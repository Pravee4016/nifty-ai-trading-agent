#!/usr/bin/env python3
"""
Detailed analysis of today's NIFTY data to identify why no signals were generated.
"""

import sys
sys.path.insert(0, '/Users/praveent/nifty-ai-trading-agent')

import pandas as pd
import logging
from data_module.fetcher import get_data_fetcher
from analysis_module.technical import TechnicalAnalyzer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

fetcher = get_data_fetcher()

# Fetch today's data
df_5m = fetcher.fetch_historical_data("NIFTY", period="1d", interval="5m")
df_15m = fetcher.fetch_historical_data("NIFTY", period="1d", interval="15m")

if df_5m is not None and not df_5m.empty:
    df_5m = fetcher.preprocess_ohlcv(df_5m)
    
    print("="*80)
    print("📊 TODAY'S PRICE ACTION ANALYSIS (Dec 1, 2025)")
    print("="*80)
    
    # Show key price levels
    print(f"\n📈 PRICE RANGE:")
    print(f"Open: {df_5m.iloc[0]['open']:.2f}")
    print(f"High: {df_5m['high'].max():.2f}")
    print(f"Low: {df_5m['low'].min():.2f}")
    print(f"Close: {df_5m.iloc[-1]['close']:.2f}")
    print(f"Change: {((df_5m.iloc[-1]['close'] - df_5m.iloc[0]['open']) / df_5m.iloc[0]['open'] * 100):.2f}%")
    
    # Show significant moves
    print(f"\n🎯 SIGNIFICANT 5M CANDLES:")
    df_5m['range_pct'] = (df_5m['high'] - df_5m['low']) / df_5m['low'] * 100
    df_5m['body_pct'] = abs(df_5m['close'] - df_5m['open']) / df_5m['open'] * 100
    
    # Check volume
    print(f"\n📊 VOLUME STATS:")
    print(f"Total Volume: {df_5m['volume'].sum()}")
    print(f"Max Volume: {df_5m['volume'].max()}")
    print(f"Avg Volume: {df_5m['volume'].mean():.2f}")
    print(f"Zero Volume Candles: {(df_5m['volume'] == 0).sum()} / {len(df_5m)}")
    
    # Find candles with large moves
    large_moves = df_5m[df_5m['range_pct'] > 0.1].copy()
    
    for idx, row in large_moves.iterrows():
        print(f"\n{idx}: O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f}")
        print(f"   Range: {row['range_pct']:.3f}% | Body: {row['body_pct']:.3f}%")
        
        # Check if this broke previous high
        prev_high = df_5m.loc[:idx].iloc[:-1]['high'].max() if len(df_5m.loc[:idx]) > 1 else 0
        if row['high'] > prev_high:
            print(f"   ✅ NEW HIGH! Broke {prev_high:.2f}")
    
    # Run technical analyzer
    print("\n" + "="*80)
    print("🔍 TECHNICAL ANALYSIS OUTPUT")
    print("="*80)
    
    analyzer = TechnicalAnalyzer("NIFTY")
    
    if df_15m is not None and not df_15m.empty:
        df_15m = fetcher.preprocess_ohlcv(df_15m)
        higher_tf_context = analyzer.get_higher_tf_context(df_15m)
        analysis = analyzer.analyze_with_multi_tf(df_5m, higher_tf_context)
        
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"Breakout Signal: {analysis.get('breakout_signal')}")
        print(f"Retest Signal: {analysis.get('retest_signal')}")
        
        # Show why no breakout was detected
        if not analysis.get('breakout_signal') or not analysis['breakout_signal'].get('signal'):
            print(f"\n⚠️  NO BREAKOUT DETECTED - Possible reasons:")
            
            # Check support/resistance levels
            sr = analyzer.calculate_support_resistance(df_5m)
            if sr:
                print(f"\n📍 Support/Resistance Levels:")
                # Access attributes directly assuming Pydantic model or dataclass
                supports = getattr(sr, 'support_levels', [])
                resistances = getattr(sr, 'resistance_levels', [])
                
                print(f"Supports: {[f'{s:.2f}' for s in supports]}")
                print(f"Resistances: {[f'{r:.2f}' for r in resistances]}")
                
                current_price = df_5m.iloc[-1]['close']
                if resistances:
                    nearest_resistance = min([r for r in resistances if r > current_price], default=None)
                    if nearest_resistance:
                        distance = ((current_price - nearest_resistance) / nearest_resistance) * 100
                        print(f"\nCurrent: {current_price:.2f}")
                        print(f"Nearest Resistance: {nearest_resistance:.2f}")
                        print(f"Distance: {distance:.2f}% {'ABOVE' if distance > 0 else 'BELOW'}")
                        print(f"Required for breakout: Price must be > {nearest_resistance:.2f}")
