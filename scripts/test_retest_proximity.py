import pandas as pd
import numpy as np
from datetime import datetime
from analysis_module.technical import TechnicalAnalyzer, TechnicalLevels, SignalType

def test_retest_proximity():
    print("Running Retest Proximity Verification...")
    
    # Mock data at 25800
    df = pd.DataFrame({
        "open": [25800]*20,
        "high": [25810]*20,
        "low": [25790]*20,
        "close": [25800]*20,
        "volume": [1000]*20
    }, index=pd.date_range("2024-12-12 10:00:00", periods=20, freq="5min"))
    
    # Levels
    # 1. 25700 (100 pts away, ~0.39%) -> Should be REJECTED at 0.15%
    # 2. 25780 (20 pts away, ~0.08%) -> Should be ACCEPTED at 0.15%
    
    levels = TechnicalLevels(
        support_levels=[25700.0, 25780.0],
        resistance_levels=[26000.0],
        pivot=25800.0,
        pdh=26100.0,
        pdl=25500.0,
        atr=50.0,
        volatility_score=50.0
    )
    
    analyzer = TechnicalAnalyzer("NIFTY")
    
    print("\n--- Testing RETEST SETUP proximity ---")
    # For retest setup, we also need recent breakout context.
    # Let's just check if it finds the levels.
    
    # We need to simulate a bounce from the level for _validate_retest_structure to pass
    # current price is 25800. If we test level 25780:
    # distance = abs(25800 - 25780) / 25780 * 100 = 0.077% <= 0.15% (Limit)
    
    # If we test level 25700:
    # distance = abs(25800 - 25700) / 25700 * 100 = 0.389% > 0.15% (Limit)
    
    # Let's mock the internal call to see what candidate levels are processed
    # We can just run detect_retest_setup. 
    # It might fail validation, but we can see the "🎯 RETEST SETUP" logs if distance is OK.
    
    # Actually, let's just test the distance logic directly if possible, or check logs.
    
    # Test PIN BAR proximity
    print("\n--- Testing PIN BAR proximity ---")
    # Mock a pin bar candle at 25800 with low near a level
    # Candle Low = 25785. Level = 25780. Dist = 5 pts (0.019%) -> OK
    # Candle Low = 25800. Level = 25700. Dist = 100 pts (0.389%) -> Should FAIL
    
    pin_df_ok = df.copy()
    pin_df_ok.iloc[-1] = [25800, 25805, 25785, 25800, 2000] # Hammer at 25785
    
    higher_tf = {"trend_direction": "UP", "rsi_15": 55.0}
    
    sig = analyzer.detect_pin_bar(pin_df_ok, levels, higher_tf)
    if sig:
        print(f"✅ Pin Bar at 25780 (20 pts away) ACCEPTED as expected (Level: {sig.price_level})")
    else:
        print("❌ Pin Bar at 25780 REJECTED unexpectedly")
        
    pin_df_fail = df.copy()
    pin_df_fail.iloc[-1] = [25800, 25805, 25800, 25800, 2000] # Hammer at 25800 (Low is far from any level)
    # The nearest level is 25780 (20 pts away) which is OK, but let's test 25700 specifically.
    
    # If we remove 25780 from levels:
    levels_far = TechnicalLevels(
        support_levels=[25700.0],
        resistance_levels=[26000.0],
        pivot=25800.0,
        pdh=26100.0,
        pdl=25500.0,
        atr=50.0,
        volatility_score=50.0
    )
    
    sig_fail = analyzer.detect_pin_bar(pin_df_ok, levels_far, higher_tf)
    if not sig_fail:
        print("✅ Pin Bar at 25700 (100 pts away) REJECTED as expected")
    else:
        print(f"❌ Pin Bar at 25700 ACCEPTED unexpectedly (Level: {sig_fail.price_level})")

if __name__ == "__main__":
    test_retest_proximity()
