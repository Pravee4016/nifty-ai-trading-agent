"""
Unit Tests for Adaptive Thresholds Module
Tests VIX-based and ATR-based RSI threshold calculations
"""

import unittest
from analysis_module.adaptive_thresholds import AdaptiveThresholds
import pandas as pd
import numpy as np


class TestAdaptiveThresholds(unittest.TestCase):
    """Test suite for adaptive threshold calculations."""
    
    def test_vix_low_volatility(self):
        """Test RSI thresholds with low VIX (< 12)."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(vix=11.0)
        
        self.assertEqual(rsi_long, 55, "Low VIX should give tighter long threshold")
        self.assertEqual(rsi_short, 45, "Low VIX should give tighter short threshold")
    
    def test_vix_normal_volatility(self):
        """Test RSI thresholds with normal VIX (12-18)."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(vix=15.0)
        
        self.assertEqual(rsi_long, 60, "Normal VIX should give default long threshold")
        self.assertEqual(rsi_short, 40, "Normal VIX should give default short threshold")
    
    def test_vix_high_volatility(self):
        """Test RSI thresholds with high VIX (> 18)."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(vix=22.0)
        
        self.assertEqual(rsi_long, 65, "High VIX should give stricter long threshold")
        self.assertEqual(rsi_short, 35, "High VIX should give stricter short threshold")
    
    def test_atr_low_percentile(self):
        """Test RSI thresholds with low ATR percentile (< 30%)."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(
            vix=None, 
            atr_percentile=25.0
        )
        
        self.assertEqual(rsi_long, 55, "Low ATR percentile should give tighter thresholds")
        self.assertEqual(rsi_short, 45)
    
    def test_atr_normal_percentile(self):
        """Test RSI thresholds with normal ATR percentile (30-70%)."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(
            vix=None,
            atr_percentile=50.0
        )
        
        self.assertEqual(rsi_long, 60, "Normal ATR percentile should give default thresholds")
        self.assertEqual(rsi_short, 40)
    
    def test_atr_high_percentile(self):
        """Test RSI thresholds with high ATR percentile (> 70%)."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(
            vix=None,
            atr_percentile=80.0
        )
        
        self.assertEqual(rsi_long, 65, "High ATR percentile should give stricter thresholds")
        self.assertEqual(rsi_short, 35)
    
    def test_vix_priority_over_atr(self):
        """Test that VIX takes priority when both available."""
        # VIX says high volatility, ATR says low
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(
            vix=20.0,  # High VIX
            atr_percentile=25.0  # Low ATR
        )
        
        # Should use VIX thresholds (65/35), not ATR (55/45)
        self.assertEqual(rsi_long, 65, "VIX should take priority over ATR")
        self.assertEqual(rsi_short, 35)
    
    def test_atr_percentile_calculation(self):
        """Test ATR percentile calculation."""
        # Create sample dataframe with ATR values
        df = pd.DataFrame({
            'atr': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190] + [200] * 50
        })
        
        current_atr = 150
        percentile = AdaptiveThresholds.calculate_atr_percentile(df, current_atr, lookback=60)
        
        # 150 is higher than first 5 values (100-140), so percentile should be ~8.3%
        self.assertLess(percentile, 20, "Percentile calculation should work correctly")
        self.assertGreater(percentile, 5)
    
    def test_atr_threshold_calculation(self):
        """Test ATR threshold as percentile."""
        # Create sample dataframe
        df = pd.DataFrame({
            'atr': [50 + i for i in range(100)]
        })
        
        atr_threshold = AdaptiveThresholds.get_atr_threshold(df, atr_period=14)
        
        # 60th percentile of 50-149 should be around 109-120
        self.assertGreater(atr_threshold, 105, "ATR threshold should be meaningful")
        self.assertLess(atr_threshold, 125)
    
    def test_market_volatile_detection(self):
        """Test volatile market detection."""
        # High VIX
        self.assertTrue(
            AdaptiveThresholds.is_market_volatile(vix=20.0),
            "VIX > 18 should be detected as volatile"
        )
        
        # High ATR percentile
        self.assertTrue(
            AdaptiveThresholds.is_market_volatile(atr_percentile=75.0),
            "ATR > 70% should be detected as volatile"
        )
        
        # Normal VIX
        self.assertFalse(
            AdaptiveThresholds.is_market_volatile(vix=15.0),
            "VIX 15 should not be volatile"
        )
    
    def test_market_choppy_detection(self):
        """Test choppy market detection."""
        # Low VIX
        self.assertTrue(
            AdaptiveThresholds.is_market_choppy(vix=11.0),
            "VIX < 12 should be detected as choppy"
        )
        
        # Low ATR percentile
        self.assertTrue(
            AdaptiveThresholds.is_market_choppy(atr_percentile=25.0),
            "ATR < 30% should be detected as choppy"
        )
        
        # Normal VIX
        self.assertFalse(
            AdaptiveThresholds.is_market_choppy(vix=15.0),
            "VIX 15 should not be choppy"
        )
    
    def test_boundary_conditions(self):
        """Test boundary conditions for VIX and ATR."""
        # VIX exactly 12
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(vix=12.0)
        self.assertEqual(rsi_long, 60, "VIX 12 should use normal thresholds")
        
        # VIX exactly 18
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(vix=18.0)
        self.assertEqual(rsi_long, 60, "VIX 18 should use normal thresholds")
        
        # ATR exactly 30%
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(atr_percentile=30.0)
        self.assertEqual(rsi_long, 60, "ATR 30% should use normal thresholds")
        
        # ATR exactly 70%
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(atr_percentile=70.0)
        self.assertEqual(rsi_long, 60, "ATR 70% should use normal thresholds")
    
    def test_none_inputs(self):
        """Test behavior when both VIX and ATR are None."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(vix=None, atr_percentile=None)
        
        # Should return default thresholds
        self.assertEqual(rsi_long, 60, "Should return default when both None")
        self.assertEqual(rsi_short, 40)


if __name__ == '__main__':
    unittest.main()
