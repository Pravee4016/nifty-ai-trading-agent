"""
Enhanced Backtest: Multiple Scenarios
Tests combo strategy with different filtering thresholds and time periods
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backtest_combo_strategy import BacktestEngine, run_backtest_comparison
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_enhanced_backtests():
    """Run backtests with multiple configurations."""
    
    print("\n" + "="*80)
    print("🚀 ENHANCED BACKTEST SUITE")
    print("="*80)
    print("\nTesting multiple scenarios:")
    print("1. Standard 2-week period")
    print("2. Extended 3-week period")
    print("3. Strict filtering (MEDIUM+ only)")
    print("\n")
    
    results = []
    
    # Scenario 1: Standard 2-week backtest
    print("="*80)
    print("📊 SCENARIO 1: Standard 2-Week Backtest")
    print("="*80)
    
    baseline_stats_1, combo_stats_1 = run_backtest_comparison()
    results.append(("2-Week Standard", baseline_stats_1, combo_stats_1))
    
    # Scenario 2: Extended 3-week period
    print("\n" + "="*80)
    print("📊 SCENARIO 2: Extended 3-Week Period")
    print("="*80)
    
    try:
        engine_baseline = BacktestEngine(use_combo=False)
        df_3week = engine_baseline.fetch_data(days=21)
        
        if df_3week is not None:
            logger.info(f"✅ Data ready: {len(df_3week)} candles")
            
            # Baseline
            engine_baseline.detect_signals(df_3week)
            engine_baseline.evaluate_outcomes(df_3week)
            baseline_stats_2 = engine_baseline.calculate_stats()
            
            # With Combo
            engine_combo = BacktestEngine(use_combo=True)
            engine_combo.detect_signals(df_3week)
            engine_combo.evaluate_outcomes(df_3week)
            combo_stats_2 = engine_combo.calculate_stats()
            
            print_comparison("3-Week", baseline_stats_2, combo_stats_2)
            results.append(("3-Week Standard", baseline_stats_2, combo_stats_2))
        else:
            logger.warning("⚠️ Couldn't fetch 3-week data")
            
    except Exception as e:
        logger.error(f"Error in Scenario 2: {e}")
    
    # Scenario 3: Strict Filtering (simulated)
    print("\n" + "="*80)
    print("📊 SCENARIO 3: Strict Filtering (MEDIUM+ Signals Only)")
    print("="*80)
    
    if combo_stats_1:
        # Simulate filtering to MEDIUM+ only
        total_original = combo_stats_1['total_signals']
        medium_signals = combo_stats_1['medium_signals']
        strong_signals = combo_stats_1['strong_signals']
        
        filtered_total = medium_signals + strong_signals
        
        # Calculate projected win rate
        if filtered_total > 0:
            medium_wins = combo_stats_1['medium_win_rate'] / 100 * medium_signals
            strong_wins = combo_stats_1['strong_win_rate'] / 100 * strong_signals if strong_signals > 0 else 0
            
            projected_wins = medium_wins + strong_wins
            projected_win_rate = (projected_wins / filtered_total * 100) if filtered_total > 0 else 0
            
            print(f"\n📊 Projected Performance with MEDIUM+ Filtering:")
            print(f"  Total Signals:     {total_original} → {filtered_total} ({filtered_total/total_original*100:.1f}%)")
            print(f"  Projected Win Rate: {combo_stats_1['win_rate']:.1f}% → {projected_win_rate:.1f}%")
            print(f"  Signal Reduction:  {(1 - filtered_total/total_original)*100:.1f}%")
            print(f"  Quality Improvement: {projected_win_rate - combo_stats_1['win_rate']:.1f}% points")
            
            # Estimate P&L improvement
            if projected_win_rate > combo_stats_1['win_rate'] * 1.2:
                print(f"\n✅ STRONG: Filtering would improve performance significantly!")
            elif projected_win_rate > combo_stats_1['win_rate']:
                print(f"\n✅ MODERATE: Filtering would improve performance")
            else:
                print(f"\n⚠️ NEUTRAL: Filtering doesn't significantly improve this dataset")
    
    # Final Summary
    print("\n" + "="*80)
    print("📊 OVERALL SUMMARY")
    print("="*80)
    
    print(f"\n{'Scenario':<25} {'Signals':<12} {'Win Rate %':<15} {'Total P&L %':<15}")
    print("-" * 67)
    
    for name, baseline, combo in results:
        if baseline and combo:
            print(f"{name:<25} B:{baseline['total_signals']:<3} C:{combo['total_signals']:<3}    "
                  f"B:{baseline['win_rate']:<6.1f} C:{combo['win_rate']:<6.1f} "
                  f"B:{baseline['total_pnl']:<6.2f} C:{combo['total_pnl']:<6.2f}")
    
    print("\n" + "="*80)
    print("✅ ENHANCED BACKTEST COMPLETE")
    print("="*80)


def print_comparison(scenario_name, baseline_stats, combo_stats):
    """Print comparison for a scenario."""
    print(f"\n📈 {scenario_name} Results:")
    print(f"  Baseline: {baseline_stats['total_signals']} signals | {baseline_stats['win_rate']:.1f}% WR | {baseline_stats['total_pnl']:.2f}% P&L")
    print(f"  Combo:    {combo_stats['total_signals']} signals | {combo_stats['win_rate']:.1f}% WR | {combo_stats['total_pnl']:.2f}% P&L")
    
    wr_change = combo_stats['win_rate'] - baseline_stats['win_rate']
    pnl_change = combo_stats['total_pnl'] - baseline_stats['total_pnl']
    
    if wr_change > 0 or pnl_change > 0:
        print(f"  Result: ✅ Combo improved by {wr_change:+.1f}% WR, {pnl_change:+.2f}% P&L")
    else:
        print(f"  Result: ➖ Similar performance ({wr_change:+.1f}% WR, {pnl_change:+.2f}% P&L)")


if __name__ == "__main__":
    run_enhanced_backtests()
