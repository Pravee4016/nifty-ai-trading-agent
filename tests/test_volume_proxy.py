
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from analysis_module.technical import TechnicalAnalyzer

class TestVolumeProxy(unittest.TestCase):
    def setUp(self):
        self.analyzer = TechnicalAnalyzer("NIFTY")
        
        # Base DataFrame setup
        self.df = pd.DataFrame({
            "open": [100.0] * 20,
            "high": [105.0] * 20,
            "low": [95.0] * 20,
            "close": [100.0] * 20,
            "volume": [1000] * 20,
            "vwap": [100.0] * 20,
            "atr": [2.0] * 20
        })

    def test_bullish_scenario(self):
        """Test a strong bullish setup (Score should be high)"""
        df = self.df.copy()
        
        # 1. VWAP Acceptance Setup (+2)
        # Positive Slope & Close > VWAP & Strong Body
        df.loc[19, "vwap"] = 102.0
        df.loc[14, "vwap"] = 101.0 # Slope +1
        df.loc[19, "open"] = 103.0
        df.loc[19, "close"] = 105.0 # Body=2, ATR=2 (Body > 0.5*ATR)
        
        # 2. Options Setup (+2)
        # Put OI Increasing (writing), Put Vol > Call Vol
        opt_data = {
            "net_put_oi_chg": 50000,
            "net_call_oi_chg": -10000,
            "total_put_vol": 100000,
            "total_call_vol": 50000
        }
        
        # 3. ATR Expansion (+1)
        df.loc[19, "atr"] = 3.0 # Avg is 2.0
        
        # 4. RVOL Setup (+1)
        df.loc[19, "volume"] = 2000 # Avg is 1000 (2.0x)
        
        result = self.analyzer.calculate_volume_proxy(df, opt_data)
        
        print("\n🔵 Bullish Test Result:", result)
        self.assertGreaterEqual(result["score"], 4)
        self.assertTrue(any("VWAP:✅Bull" in x for x in result["breakdown"]))
        self.assertTrue(any("Opt:✅Bull" in x for x in result["breakdown"]))

    def test_bearish_scenario(self):
        """Test a strong bearish setup"""
        df = self.df.copy()
        
        # 1. VWAP Acceptance Setup (+2)
        # Negative Slope & Close < VWAP
        df.loc[19, "vwap"] = 100.0
        df.loc[14, "vwap"] = 101.0 # Slope -1
        df.loc[19, "open"] = 99.0
        df.loc[19, "close"] = 97.0 # Close < VWAP
        
        # 2. Options Setup (+2)
        # Call OI Increasing (writing), Call Vol > Put Vol
        opt_data = {
            "net_call_oi_chg": 50000,
            "net_put_oi_chg": -10000,
            "total_call_vol": 100000,
            "total_put_vol": 50000
        }
        
        result = self.analyzer.calculate_volume_proxy(df, opt_data)
        print("\n🔴 Bearish Test Result:", result)
        self.assertGreaterEqual(result["score"], 4)
        self.assertTrue(any("VWAP:✅Bear" in x for x in result["breakdown"]))

    def test_neutral_chop_case(self):
        """Test a chop scenario (Score should be low)"""
        df = self.df.copy()
        
        # 1. VWAP Flat & Weak Body
        df.loc[19, "vwap"] = 100.0
        df.loc[14, "vwap"] = 100.0
        df.loc[19, "open"] = 100.1
        df.loc[19, "close"] = 100.2 # Tiny body < 0.5 ATR
        
        # 2. Options Mixed
        opt_data = {
            "net_call_oi_chg": 0,
            "net_put_oi_chg": 0,
            "total_call_vol": 1000,
            "total_put_vol": 1000
        }
        
        result = self.analyzer.calculate_volume_proxy(df, opt_data)
        print("\n⚪ Neutral Test Result:", result)
        self.assertLess(result["score"], 4)

if __name__ == "__main__":
    unittest.main()
