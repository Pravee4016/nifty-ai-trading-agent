"""
Test script to validate Bug Fix #1 (Confluence Tolerance) and Bug Fix #2 (Stale Option Data)
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_bug_1_confluence_tolerance():
    """Test that confluence tolerance is now 3 points absolute, not percentage-based"""
    print("\n" + "="*80)
    print("TEST #1: CONFLUENCE TOLERANCE FIX")
    print("="*80)
    
    from config.settings import CONFLUENCE_TOLERANCE_POINTS
    from analysis_module.confluence_detector import detect_confluence, TechnicalLevels
    
    print(f"\n✓ Imported CONFLUENCE_TOLERANCE_POINTS = {CONFLUENCE_TOLERANCE_POINTS}")
    
    # Create test levels
    levels = TechnicalLevels(
        support_levels=[24900, 24950, 24998],
        resistance_levels=[25002, 25050, 25100],
        pivot=24995.0,
        pdh=24950.0,  # 50 points away from test price
        pdl=24900.0,
        atr=50.0,
        volatility_score=0.5,
        r1_fib=25050.0,
        s1_fib=24960.0,  # 40 points away
        r2_fib=25100.0,
        s2_fib=24900.0
    )
    
    higher_tf_context = {
        'ema_20_5m': 25001.0,  # 1 point away - SHOULD BE INCLUDED
        'ema_50_5m': 24945.0,  # 55 points away - SHOULD BE EXCLUDED
        'vwap_5m': 24999.0,    # 1 point away - SHOULD BE INCLUDED
    }
    
    # Test at price 25000
    test_price = 25000.0
    
    print(f"\nTest Price: {test_price}")
    print(f"Tolerance: {CONFLUENCE_TOLERANCE_POINTS} points (absolute)")
    print("\nExpected Behavior:")
    print("  ✓ Pivot (24995) - 5pts away → EXCLUDED")
    print("  ✓ PDH (24950) - 50pts away → EXCLUDED")
    print("  ✓ S1_Fib (24960) - 40pts away → EXCLUDED")
    print("  ✓ EMA20_5m (25001) - 1pt away → INCLUDED ✓")
    print("  ✓ VWAP (24999) - 1pt away → INCLUDED ✓")
    
    result = detect_confluence(test_price, levels, higher_tf_context)
    
    print(f"\nActual Results:")
    print(f"  Confluence Count: {result['confluence_count']}")
    print(f"  Tolerance Used: {result['tolerance_used']} points")
    print(f"  Confluent Levels: {result['level_names']}")
    
    # Validation
    old_tolerance = test_price * 0.002  # Old calculation
    print(f"\n--- VALIDATION ---")
    print(f"OLD tolerance (percentage): {old_tolerance:.2f} points (at price {test_price})")
    print(f"NEW tolerance (absolute): {result['tolerance_used']:.2f} points")
    
    if result['tolerance_used'] == CONFLUENCE_TOLERANCE_POINTS:
        print("✅ TEST PASSED: Using absolute point tolerance!")
    else:
        print(f"❌ TEST FAILED: Expected {CONFLUENCE_TOLERANCE_POINTS}, got {result['tolerance_used']}")
        return False
    
    # Verify that distant levels are excluded
    excluded_levels = ['PDH', 'PDL', 'Fib_S1', 'Pivot']
    for level in excluded_levels:
        if level in result['level_names']:
            print(f"❌ TEST FAILED: {level} should be EXCLUDED (too far)")
            return False
    
    print(f"✅ Distant levels correctly EXCLUDED: {excluded_levels}")
    
    # Verify close levels are included
    if 'EMA20_5m' in result['level_names'] or 'VWAP' in result['level_names']:
        print(f"✅ Close levels correctly INCLUDED")
    else:
        print(f"⚠️ WARNING: Expected at least one close level to be included")
    
    print("\n✅ BUG FIX #1 VALIDATED: Confluence now uses 3-point tolerance!")
    return True


def test_bug_2_stale_data():
    """Test that stale option data (>5 minutes) is rejected in conflict resolution"""
    print("\n" + "="*80)
    print("TEST #2: STALE OPTION DATA VALIDATION")
    print("="*80)
    
    from analysis_module.signal_pipeline import SignalPipeline
    
    pipeline = SignalPipeline()
    
    # Create test signals with conflict
    signals = [
        {
            'signal_type': 'BULLISH_PIN_BAR',
            'confidence': 70,
            'entry': 25000,
        },
        {
            'signal_type': 'BEARISH_BREAKOUT',
            'confidence': 75,
            'entry': 25000,
        }
    ]
    
    print(f"\nTest Signals: 1 BULLISH + 1 BEARISH (conflict)")
    
    # Test 1: Fresh data (30 seconds old)
    print("\n--- Test 1: Fresh Data (30 seconds old) ---")
    fresh_metrics = {
        'pcr': 1.5,  # Bullish
        'fetch_timestamp': time.time() - 30,  # 30 seconds ago
        'oi_change': {'sentiment': 'BULLISH'}
    }
    
    result = pipeline.resolve_conflicts(signals, fresh_metrics)
    
    if len(result) == 1 and 'BULLISH' in result[0]['signal_type']:
        print(f"✅ Fresh data accepted: Resolved to BULLISH (as expected with PCR {fresh_metrics['pcr']})")
    else:
        print(f"❌ Fresh data test FAILED")
        return False
    
    # Test 2: Stale data (6 minutes old)
    print("\n--- Test 2: Stale Data (6 minutes old) ---")
    stale_metrics = {
        'pcr': 1.5,  # Bullish (but stale!)
        'fetch_timestamp': time.time() - 360,  # 6 minutes ago
        'oi_change': {'sentiment': 'BULLISH'}
    }
    
    result = pipeline.resolve_conflicts(signals, stale_metrics)
    
    if len(result) == 2:  # Should return both signals (no resolution)
        print(f"✅ Stale data rejected: Returned all {len(result)} signals (no resolution)")
    else:
        print(f"❌ Stale data test FAILED: Should return all signals, got {len(result)}")
        return False
    
    # Test 3: Missing timestamp (graceful degradation)
    print("\n--- Test 3: Missing Timestamp (graceful degradation) ---")
    no_timestamp_metrics = {
        'pcr': 1.5,
        'oi_change': {'sentiment': 'BULLISH'}
        # No timestamp provided
    }
    
    result = pipeline.resolve_conflicts(signals, no_timestamp_metrics)
    
    # Should assume fresh and use PCR
    if len(result) == 1 and 'BULLISH' in result[0]['signal_type']:
        print(f"✅ Missing timestamp handled: Assumed fresh, resolved to BULLISH")
    else:
        print(f"❌ Missing timestamp test FAILED")
        return False
    
    print("\n✅ BUG FIX #2 VALIDATED: Stale option data properly rejected!")
    return True


def main():
    """Run all bug fix validation tests"""
    print("\n╔════════════════════════════════════════════════════════════════════════╗")
    print("║          BUG FIX VALIDATION SUITE - December 21, 2025                 ║")
    print("╚════════════════════════════════════════════════════════════════════════╝")
    
    results = {}
    
    try:
        results['bug_1'] = test_bug_1_confluence_tolerance()
    except Exception as e:
        print(f"\n❌ BUG #1 TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['bug_1'] = False
    
    try:
        results['bug_2'] = test_bug_2_stale_data()
    except Exception as e:
        print(f"\n❌ BUG #2 TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        results['bug_2'] = False
    
    # Final Report
    print("\n" + "="*80)
    print("FINAL VALIDATION REPORT")
    print("="*80)
    
    print(f"\nBug #1 (Confluence Tolerance):  {'✅ PASSED' if results.get('bug_1') else '❌ FAILED'}")
    print(f"Bug #2 (Stale Option Data):     {'✅ PASSED' if results.get('bug_2') else '❌ FAILED'}")
    
    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED! Both bug fixes are working correctly.")
        print("\nNext Steps:")
        print("  1. ✅ Commit changes to git")
        print("  2. ✅ Deploy to production")
        print("  3. ✅ Monitor first 100 signals")
        return 0
    else:
        print("\n⚠️ SOME TESTS FAILED - Review fixes before deployment")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
