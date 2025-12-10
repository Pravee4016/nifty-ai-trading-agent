
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from analysis_module.manipulation_guard import CircuitBreaker

class TestCircuitBreaker(unittest.TestCase):
    def setUp(self):
        self.breaker = CircuitBreaker()
        self.columns = ['open', 'high', 'low', 'close', 'volume']
        
    def create_candle(self, open_p, high, low, close):
        return pd.DataFrame([[open_p, high, low, close, 1000]], columns=self.columns)

    def test_normal_market(self):
        """Test with normal low volatility data"""
        df = self.create_candle(26000, 26010, 25990, 26005)
        is_safe, reason = self.breaker.check_market_integrity(df, 26005)
        self.assertTrue(is_safe)
        self.assertEqual(reason, "Market Normal")

    def test_flash_crash_trigger(self):
        """Test 0.5% drop in single 5m candle"""
        # 0.5% of 26000 is 130 points.
        # Candle Limit is MAX_1MIN_MOVE_PCT * 2.5 = 0.4 * 2.5 = 1.0% actually in my implementation?
        # Let's check logic: if move_pct > (MAX_1MIN_MOVE_PCT * 2.5):
        # 0.4 * 2.5 = 1.0%
        
        # Let's create a huge 1.2% drop candle
        # Open: 26000, Low: 25600 (400 pts drop ~ 1.5%)
        df = self.create_candle(26000, 26010, 25600, 25650)
        
        is_safe, reason = self.breaker.check_market_integrity(df, 25650)
        self.assertFalse(is_safe)
        self.assertIn("Flash Crash Protection", reason)
        self.assertTrue(self.breaker.triggered)

    @patch('analysis_module.manipulation_guard.datetime')
    def test_expiry_gamma_guard_active(self, mock_datetime):
        """Test Expiry Guard on Tuesday afternoon"""
        # Mock time to Tuesday (weekday 1) at 14:30
        # Year 2025, Dec 2 was Tuesday
        mock_now = datetime(2025, 12, 2, 14, 30, 0)
        mock_datetime.now.return_value = mock_now
        
        df = self.create_candle(26000, 26010, 25990, 26005)
        
        # Should return False due to Time
        is_safe, reason = self.breaker.check_market_integrity(df, 26005, "NIFTY 50")
        
        self.assertFalse(is_safe)
        self.assertIn("Gamma Guard Active", reason)

    @patch('analysis_module.manipulation_guard.datetime')
    def test_expiry_gamma_guard_inactive_morning(self, mock_datetime):
        """Test Expiry Guard on Tuesday morning (Should be Safe)"""
        # Tuesday 10:00 AM
        mock_now = datetime(2025, 12, 2, 10, 00, 0)
        mock_datetime.now.return_value = mock_now
        
        df = self.create_candle(26000, 26010, 25990, 26005)
        
        is_safe, reason = self.breaker.check_market_integrity(df, 26005, "NIFTY 50")
        self.assertTrue(is_safe)

    @patch('analysis_module.manipulation_guard.datetime')
    def test_expiry_gamma_guard_inactive_monday(self, mock_datetime):
        """Test Expiry Guard on Monday (Should be Safe)"""
        # Monday (weekday 0) at 14:30
        mock_now = datetime(2025, 12, 1, 14, 30, 0)
        mock_datetime.now.return_value = mock_now
        
        df = self.create_candle(26000, 26010, 25990, 26005)
        
        is_safe, reason = self.breaker.check_market_integrity(df, 26005, "NIFTY 50")
        self.assertTrue(is_safe)

if __name__ == '__main__':
    unittest.main()
