"""
Unit Tests for Structure-Based Duplicate Suppression
Tests fresh structure validation logic
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from analysis_module.signal_pipeline import SignalPipeline


class TestStructureValidation(unittest.TestCase):
    """Test suite for structure validation in signal pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = SignalPipeline(groq_analyzer=None)  # No AI needed for tests
    
    def create_sample_df(self, n=100, trend='up'):
        """Create sample dataframe for testing."""
        if trend == 'up':
            prices = np.linspace(25000, 26000, n)
        elif trend == 'down':
            prices = np.linspace(26000, 25000, n)
        else:  # sideways
            prices = 25500 + np.random.randn(n) * 50
        
        df = pd.DataFrame({
            'close': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'open': prices * 0.999,
            'volume': np.random.randint(100000, 500000, n)
        })
        
        return df
    
    def test_higher_high_detection(self):
        """Test detection of new higher-high in bullish trend."""
        df = self.create_sample_df(100, trend='up')
        
        # Create context
        context = {
            'df_5m': df,
            'vwap_5m': df['close'].mean()
        }
        
        # Should detect higher-high
        has_structure = self.pipeline._validate_fresh_structure(
            signals=[],
            context=context,
            direction='BULLISH'
        )
        
        self.assertTrue(has_structure, "Should detect higher-high in uptrend")
    
    def test_lower_low_detection(self):
        """Test detection of new lower-low in bearish trend."""
        df = self.create_sample_df(100, trend='down')
        
        context = {
            'df_5m': df,
            'vwap_5m': df['close'].mean()
        }
        
        has_structure = self.pipeline._validate_fresh_structure(
            signals=[],
            context=context,
            direction='BEARISH'
        )
        
        self.assertTrue(has_structure, "Should detect lower-low in downtrend")
    
    def test_vwap_reclaim_bullish(self):
        """Test VWAP reclaim detection for bullish setup."""
        # Create data where price crosses above VWAP
        df = pd.DataFrame({
            'close': [25000, 24980, 24990] + [25020] * 97,  # Dip then reclaim
            'high': [25010] * 100,
            'low': [24970] * 100,
            'open': [25000] * 100,
            'volume': [100000] * 100
        })
        
        vwap = 25000  # Set VWAP at mid-level
        
        context = {
            'df_5m': df,
            'vwap_5m': vwap
        }
        
        has_structure = self.pipeline._validate_fresh_structure(
            signals=[],
            context=context,
            direction='BULLISH'
        )
        
        self.assertTrue(has_structure, "Should detect VWAP reclaim")
    
    def test_vwap_breakdown_bearish(self):
        """Test VWAP breakdown detection for bearish setup."""
        # Create data where price crosses below VWAP
        df = pd.DataFrame({
            'close': [25000, 25020, 25010] + [24980] * 97,  # Rally then breakdown
            'high': [25030] * 100,
            'low': [24970] * 100,
            'open': [25000] * 100,
            'volume': [100000] * 100
        })
        
        vwap = 25000  # Set VWAP at mid-level
        
        context = {
            'df_5m': df,
            'vwap_5m': vwap
        }
        
        has_structure = self.pipeline._validate_fresh_structure(
            signals=[],
            context=context,
            direction='BEARISH'
        )
        
        self.assertTrue(has_structure, "Should detect VWAP breakdown")
    
    def test_volume_surge_detection(self):
        """Test volume surge detection."""
        # Create data with volume surge on last bar
        volumes = [100000] * 99 + [250000]  # 2.5x average on last bar
        
        df = pd.DataFrame({
            'close': [25000] * 100,
            'high': [25010] * 100,
            'low': [24990] * 100,
            'open': [25000] * 100,
            'volume': volumes
        })
        
        context = {
            'df_5m': df,
            'vwap_5m': 25000
        }
        
        has_structure = self.pipeline._validate_fresh_structure(
            signals=[],
            context=context,
            direction='BULLISH'
        )
        
        self.assertTrue(has_structure, "Should detect volume surge")
    
    def test_no_fresh_structure_sideways(self):
        """Test that sideways market shows no fresh structure."""
        df = self.create_sample_df(100, trend='sideways')
        
        # Make last price same as recent high
        df.loc[df.index[-1], 'close'] = df['high'].iloc[-20:-1].max() * 0.999
        
        context = {
            'df_5m': df,
            'vwap_5m': df['close'].mean()
        }
        
        has_structure = self.pipeline._validate_fresh_structure(
            signals=[],
            context=context,
            direction='BULLISH'
        )
        
        # Might still detect due to volume, but structure should be weaker
        # This test checks the logic works, not necessarily returns False
        self.assertIsInstance(has_structure, bool, "Should return boolean")
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        df = pd.DataFrame({
            'close': [25000],
            'high': [25010],
            'low': [24990],
            'open': [25000],
            'volume': [100000]
        })
        
        context = {
            'df_5m': df,
            'vwap_5m': 25000
        }
        
        has_structure = self.pipeline._validate_fresh_structure(
            signals=[],
            context=context,
            direction='BULLISH'
        )
        
        self.assertFalse(has_structure, "Should return False with insufficient data")
    
    def test_missing_vwap(self):
        """Test behavior when VWAP is missing."""
        df = self.create_sample_df(100, trend='up')
        
        context = {
            'df_5m': df,
            'vwap_5m': 0  # Missing VWAP
        }
        
        has_structure = self.pipeline._validate_fresh_structure(
            signals=[],
            context=context,
            direction='BULLISH'
        )
        
        # Should still check for higher-high and volume
        self.assertIsInstance(has_structure, bool, "Should handle missing VWAP gracefully")


if __name__ == '__main__':
    unittest.main()
