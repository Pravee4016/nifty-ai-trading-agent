"""
Unit Tests for Tick Size Calculations
Tests get_tick_size function and target rounding
"""

import unittest
from config.settings import get_tick_size, TICK_SIZE_MAP


class TestTickSize(unittest.TestCase):
    """Test suite for tick size calculations."""
    
    def test_nifty_spot_tick_size(self):
        """Test NIFTY spot uses 1.0 tick size."""
        tick_size = get_tick_size("NIFTY", is_option=False)
        self.assertEqual(tick_size, 1.0, "NIFTY spot should have 1.0 tick size")
    
    def test_nifty_option_tick_size(self):
        """Test NIFTY option uses 0.05 tick size."""
        tick_size = get_tick_size("NIFTY", is_option=True)
        self.assertEqual(tick_size, 0.05, "NIFTY option should have 0.05 tick size")
    
    def test_banknifty_spot_tick_size(self):
        """Test BANKNIFTY spot uses 1.0 tick size."""
        tick_size = get_tick_size("BANKNIFTY", is_option=False)
        self.assertEqual(tick_size, 1.0, "BANKNIFTY spot should have 1.0 tick size")
    
    def test_banknifty_option_tick_size(self):
        """Test BANKNIFTY option uses 0.05 tick size."""
        tick_size = get_tick_size("BANKNIFTY", is_option=True)
        self.assertEqual(tick_size, 0.05, "BANKNIFTY option should have 0.05 tick size")
    
    def test_finnifty_spot_tick_size(self):
        """Test FINNIFTY spot uses 1.0 tick size."""
        tick_size = get_tick_size("FINNIFTY", is_option=False)
        self.assertEqual(tick_size, 1.0, "FINNIFTY spot should have 1.0 tick size")
    
    def test_unknown_instrument_fallback(self):
        """Test unknown instrument uses default 1.0 tick size."""
        tick_size = get_tick_size("UNKNOWN_INSTRUMENT", is_option=False)
        self.assertEqual(tick_size, 1.0, "Unknown instrument should default to 1.0")
    
    def test_price_rounding_spot(self):
        """Test price rounding for NIFTY spot."""
        tick_size = get_tick_size("NIFTY", is_option=False)
        
        # Test various prices
        price = 25915.35
        rounded = round(price / tick_size) * tick_size
        self.assertEqual(rounded, 25915.0, "Should round to whole number")
        
        price = 25915.75
        rounded = round(price / tick_size) * tick_size
        self.assertEqual(rounded, 25916.0, "Should round up to whole number")
    
    def test_price_rounding_option(self):
        """Test price rounding for NIFTY option."""
        tick_size = get_tick_size("NIFTY", is_option=True)
        
        # Test various prices
        price = 123.47
        rounded = round(price / tick_size) * tick_size
        self.assertAlmostEqual(rounded, 123.45, places=2, msg="Should round to 0.05 increment")
        
        price = 123.48
        rounded = round(price / tick_size) * tick_size
        self.assertAlmostEqual(rounded, 123.50, places=2, msg="Should round to 0.05 increment")
    
    def test_target_calculation_accuracy(self):
        """Test that targets are correctly rounded for spot."""
        tick_size = get_tick_size("NIFTY", is_option=False)
        
        # Simulate R:R calculation
        entry = 25900.0
        sl = 25880.0
        risk = entry - sl  # 20 points
        
        # T1 at 1.5R
        t1_raw = entry + (risk * 1.5)  # 25930.0
        t1_rounded = round(t1_raw / tick_size) * tick_size
        
        self.assertEqual(t1_rounded, 25930.0, "T1 should be whole number")
        self.assertTrue(t1_rounded % 1 == 0, "T1 should have no decimals")
    
    def test_risk_reward_accuracy(self):
        """Test that R:R ratio is preserved after rounding."""
        tick_size = get_tick_size("NIFTY", is_option=False)
        
        entry = 25900.0
        sl = 25890.0
        risk = abs(entry - sl)  # 10 points
        
        # T1 at 1:1.5 R:R
        t1 = round((entry + risk * 1.5) / tick_size) * tick_size
        actual_rr = abs(t1 - entry) / risk
        
        self.assertAlmostEqual(actual_rr, 1.5, places=1, msg="R:R should be preserved")
    
    def test_tick_size_map_completeness(self):
        """Test that TICK_SIZE_MAP has all required entries."""
        required_instruments = [
            "NIFTY_SPOT", "NIFTY_OPTION",
            "BANKNIFTY_SPOT", "BANKNIFTY_OPTION",
            "FINNIFTY_SPOT", "FINNIFTY_OPTION"
        ]
        
        for inst in required_instruments:
            self.assertIn(inst, TICK_SIZE_MAP, f"{inst} should be in TICK_SIZE_MAP")
    
    def test_tick_size_values(self):
        """Test that tick sizes have correct values."""
        # Spot should be 1.0
        self.assertEqual(TICK_SIZE_MAP["NIFTY_SPOT"], 1.0)
        self.assertEqual(TICK_SIZE_MAP["BANKNIFTY_SPOT"], 1.0)
        self.assertEqual(TICK_SIZE_MAP["FINNIFTY_SPOT"], 1.0)
        
        # Options should be 0.05
        self.assertEqual(TICK_SIZE_MAP["NIFTY_OPTION"], 0.05)
        self.assertEqual(TICK_SIZE_MAP["BANKNIFTY_OPTION"], 0.05)
        self.assertEqual(TICK_SIZE_MAP["FINNIFTY_OPTION"], 0.05)


if __name__ == '__main__':
    unittest.main()
