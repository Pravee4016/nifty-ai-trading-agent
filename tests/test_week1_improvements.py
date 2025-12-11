"""
Unit tests for Week 1 Signal Quality Improvements
Tests: Retest validation, Target capping, RVOL filter
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis_module.technical import TechnicalAnalyzer, TechnicalLevels


class TestRetestValidation:
    """Test retest structure validation logic"""
    
    def create_dummy_candle_data(self, num_candles=20):
        """Create dummy OHLCV data"""
        dates = pd.date_range(end=datetime.now(), periods=num_candles, freq='5min')
        base_price = 25000
        
        data = {
            'open': [base_price + np.random.randint(-50, 50) for _ in range(num_candles)],
            'high': [base_price + np.random.randint(0, 100) for _ in range(num_candles)],
            'low': [base_price - np.random.randint(0, 100) for _ in range(num_candles)],
            'close': [base_price + np.random.randint(-50, 50) for _ in range(num_candles)],
            'volume': [100000 + np.random.randint(-20000, 20000) for _ in range(num_candles)],
        }
        
        df = pd.DataFrame(data, index=dates)
        return df
    
    def test_valid_bounce(self):
        """Test: Valid bounce should pass validation"""
        print("\n📝 Test 1: Valid Bounce")
        
        analyzer = TechnicalAnalyzer("NIFTY")
        df = self.create_dummy_candle_data(20)
        
        # Create a valid bounce candle
        # Low touches 25000, close is 50% up from low
        df.iloc[-1]['low'] = 25000
        df.iloc[-1]['high'] = 25060
        df.iloc[-1]['close'] = 25030  # 50% from low (valid)
        df.iloc[-1]['volume'] = 120000  # Good volume
        
        level = 25000
        is_valid, reason = analyzer._validate_retest_structure(df, level, "LONG")
        
        print(f"   Result: {is_valid}")
        print(f"   Reason: {reason}")
        assert is_valid == True, "Valid bounce should pass"
        print("   ✅ PASSED")
    
    def test_weak_bounce(self):
        """Test: Weak bounce should fail validation"""
        print("\n📝 Test 2: Weak Bounce (should reject)")
        
        analyzer = TechnicalAnalyzer("NIFTY")
        df = self.create_dummy_candle_data(20)
        
        # Create weak bounce: close near low
        df.iloc[-1]['low'] = 25000
        df.iloc[-1]['high'] = 25060
        df.iloc[-1]['close'] = 25010  # Only 17% from low (weak)
        df.iloc[-1]['volume'] = 120000
        
        level = 25000
        is_valid, reason = analyzer._validate_retest_structure(df, level, "LONG")
        
        print(f"   Result: {is_valid}")
        print(f"   Reason: {reason}")
        assert is_valid == False, "Weak bounce should fail"
        print("   ✅ PASSED")
    
    def test_low_volume(self):
        """Test: Low volume should fail validation"""
        print("\n📝 Test 3: Low Volume (should reject)")
        
        analyzer = TechnicalAnalyzer("NIFTY")
        df = self.create_dummy_candle_data(20)
        
        # Good bounce but low volume
        df.iloc[-1]['low'] = 25000
        df.iloc[-1]['high'] = 25060
        df.iloc[-1]['close'] = 25030  # Good bounce
        df.iloc[-1]['volume'] = 50000  # Low volume (< 70% of avg)
        
        level = 25000
        is_valid, reason = analyzer._validate_retest_structure(df, level, "LONG")
        
        print(f"   Result: {is_valid}")
        print(f"   Reason: {reason}")
        assert is_valid == False, "Low volume should fail"
        print("   ✅ PASSED")


class TestTargetCapping:
    """Test TP3 capping at 3x risk"""
    
    def test_target_cap_long(self):
        """Test: TP3 uses 3x cap when no resistance levels available"""
        print("\n📝 Test 4: TP3 Capping (LONG)")
        
        analyzer = TechnicalAnalyzer("NIFTY")
        
        # Create dummy support/resistance with NO levels above entry
        # This forces the code to use calculated 3x cap
        levels = TechnicalLevels(
            support_levels=[24900, 24800, 24700],
            resistance_levels=[],  # No resistance levels - will use 3x calc
            pivot=25000,
            pdh=25150,
            pdl=24850,
            atr=50,
            volatility_score=0.5
        )
        
        entry = 25000
        sl = 24950  # 50 pts risk
        
        tp1, tp2, tp3 = analyzer._calculate_multi_targets(
            entry, sl, "LONG", levels, is_strong_trend=True
        )
        
        max_expected_tp3 = entry + (50 * 3.0)  # 3x cap = 25150
        
        print(f"   Entry: {entry}, SL: {sl}, Risk: 50 pts")
        print(f"   TP1: {tp1}")
        print(f"   TP2: {tp2}")
        print(f"   TP3: {tp3}")
        print(f"   Expected TP3 (3x): {max_expected_tp3}")
        
        # When no resistance levels, TP3 should be 3x risk
        assert tp3 == max_expected_tp3, f"TP3 {tp3} should be {max_expected_tp3}"
        print(f"   ✅ PASSED (TP3 = {tp3}, correctly capped at 3x)")
    
    def test_target_cap_short(self):
        """Test: TP3 uses 3x cap when no support levels available"""
        print("\n📝 Test 5: TP3 Capping (SHORT)")
        
        analyzer = TechnicalAnalyzer("NIFTY")
        
        # No support levels below entry - forces 3x calc
        levels = TechnicalLevels(
            support_levels=[],  # No support levels - will use 3x calc
            resistance_levels=[25100, 25200, 25300],
            pivot=25000,
            pdh=25150,
            pdl=24850,
            atr=50,
            volatility_score=0.5
        )
        
        entry = 25000
        sl = 25050  # 50 pts risk
        
        tp1, tp2, tp3 = analyzer._calculate_multi_targets(
            entry, sl, "SHORT", levels, is_strong_trend=True
        )
        
        min_expected_tp3 = entry - (50 * 3.0)  # 3x cap = 24850
        
        print(f"   Entry: {entry}, SL: {sl}, Risk: 50 pts")
        print(f"   TP1: {tp1}")
        print(f"   TP2: {tp2}")
        print(f"   TP3: {tp3}")
        print(f"   Expected TP3 (3x): {min_expected_tp3}")
        
        # When no support levels, TP3 should be 3x risk
        assert tp3 == min_expected_tp3, f"TP3 {tp3} should be {min_expected_tp3}"
        print(f"   ✅ PASSED (TP3 = {tp3}, correctly capped at 3x)")


class TestRVOL:
    """Test RVOL calculation"""
    
    def create_volume_data(self, volumes):
        """Create dummy data with specific volumes"""
        dates = pd.date_range(end=datetime.now(), periods=len(volumes), freq='5min')
        base_price = 25000
        
        data = {
            'open': [base_price] * len(volumes),
            'high': [base_price + 20] * len(volumes),
            'low': [base_price - 20] * len(volumes),
            'close': [base_price] * len(volumes),
            'volume': volumes,
        }
        
        return pd.DataFrame(data, index=dates)
    
    def test_rvol_high(self):
        """Test: High RVOL (2x average)"""
        print("\n📝 Test 6: RVOL Calculation (High Volume)")
        
        analyzer = TechnicalAnalyzer("NIFTY")
        
        # 20 candles with 100K volume, last candle 200K
        volumes = [100000] * 20
        volumes[-1] = 200000  # 2x average
        
        df = self.create_volume_data(volumes)
        
        rvol, desc = analyzer._calculate_rvol(df)
        
        print(f"   {desc}")
        print(f"   RVOL: {rvol:.2f}x")
        
        assert rvol >= 1.9, f"RVOL should be ~2.0x, got {rvol}"
        print("   ✅ PASSED (RVOL correctly calculated)")
    
    def test_rvol_low(self):
        """Test: Low RVOL (0.5x average)"""
        print("\n📝 Test 7: RVOL Calculation (Low Volume)")
        
        analyzer = TechnicalAnalyzer("NIFTY")
        
        # 20 candles with 100K volume, last candle 50K
        volumes = [100000] * 20
        volumes[-1] = 50000  # 0.5x average
        
        df = self.create_volume_data(volumes)
        
        rvol, desc = analyzer._calculate_rvol(df)
        
        print(f"   {desc}")
        print(f"   RVOL: {rvol:.2f}x")
        
        assert rvol <= 0.6, f"RVOL should be ~0.5x, got {rvol}"
        print("   ✅ PASSED (Low RVOL detected)")


def run_all_tests():
    """Run all unit tests"""
    print("="*60)
    print("🧪 Week 1 Improvements - Unit Tests")
    print("="*60)
    
    # Retest validation tests
    print("\n" + "="*60)
    print("📋 Retest Structure Validation Tests")
    print("="*60)
    retest_tests = TestRetestValidation()
    retest_tests.test_valid_bounce()
    retest_tests.test_weak_bounce()
    retest_tests.test_low_volume()
    
    # Target capping tests
    print("\n" + "="*60)
    print("📋 Target Capping Tests (3x Risk)")
    print("="*60)
    target_tests = TestTargetCapping()
    target_tests.test_target_cap_long()
    target_tests.test_target_cap_short()
    
    # RVOL tests
    print("\n" + "="*60)
    print("📋 RVOL Calculation Tests")
    print("="*60)
    rvol_tests = TestRVOL()
    rvol_tests.test_rvol_high()
    rvol_tests.test_rvol_low()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\n📊 Summary:")
    print("   - Retest validation: 3/3 tests passed")
    print("   - Target capping: 2/2 tests passed")
    print("   - RVOL calculation: 2/2 tests passed")
    print("\n   Total: 7/7 tests passed ✅")
    print("="*60)


if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
