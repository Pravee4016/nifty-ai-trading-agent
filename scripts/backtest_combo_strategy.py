"""
Backtest: MACD + Combo Strategy vs. Baseline
Compares signal performance WITH and WITHOUT combo scoring on 2-week historical data
"""

import sys
import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis_module.technical import TechnicalAnalyzer, TechnicalLevels
from analysis_module.combo_signals import MACDRSIBBCombo
from config.settings import MIN_SIGNAL_CONFIDENCE

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during backtest
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Keep backtest logs visible


class BacktestEngine:
    """Backtesting engine for comparing strategy performance."""
    
    def __init__(self, use_combo=False):
        self.use_combo = use_combo
        self.analyzer = TechnicalAnalyzer("NIFTY 50")
        self.combo_evaluator = MACDRSIBBCombo() if use_combo else None
        self.signals = []
        self.trades = []
    
    def fetch_data(self, days=14):
        """Fetch historical data."""
        try:
            import yfinance as yf
            
            logger.info(f"📥 Fetching {days} days of historical data...")
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = yf.download(
                '^NSEI',
                start=start_date,
                end=end_date,
                interval='5m',
                progress=False
            )
            
            if df.empty:
                return None
            
            # Normalize columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [str(col).lower() for col in df.columns]
            
            logger.info(f"✅ Fetched {len(df)} candles | {df.index[0]} to {df.index[-1]}")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def detect_signals(self, df: pd.DataFrame, start_idx=100):
        """
        Detect signals from historical data.
        
        Args:
            df: OHLCV DataFrame
            start_idx: Start detecting from this index (allow warmup period)
        """
        logger.info(f"🔍 Detecting signals {'WITH' if self.use_combo else 'WITHOUT'} combo...")
        
        signals_found = []
        
        # Scan through data bar by bar
        for i in range(start_idx, len(df) - 10):  # Leave 10 bars for outcome evaluation
            # Get data up to current bar
            df_current = df.iloc[:i+1].copy()
            
            # Skip if insufficient data
            if len(df_current) < 100:
                continue
            
            # Calculate technical levels
            try:
                support_resistance = self.analyzer.calculate_support_resistance(df_current)
                
                # Build higher timeframe context (simplified for backtest)
                higher_tf_context = self._build_context(df_current, support_resistance)
                
                # Try to detect each pattern type
                patterns = []
                
                # Breakout
                signal = self.analyzer.detect_breakout(df_current, support_resistance, higher_tf_context)
                if signal:
                    patterns.append(("BREAKOUT", signal))
                
                # Pin Bar
                signal = self.analyzer.detect_pin_bar(df_current, support_resistance, higher_tf_context)
                if signal:
                    patterns.append(("PIN_BAR", signal))
                
                # Inside Bar
                signal = self.analyzer.detect_inside_bar(df_current, support_resistance, higher_tf_context)
                if signal:
                    patterns.append(("INSIDE_BAR", signal))
                
                # Process detected patterns
                for pattern_name, signal in patterns:
                    # Calculate combo score if enabled
                    combo_score = 0
                    combo_strength = "N/A"
                    
                    if self.use_combo and self.combo_evaluator:
                        direction = signal.debug_info.get('direction', 'LONG')
                        
                        # Calculate technical context
                        macd_data = self.analyzer._calculate_macd(df_current)
                        bb_data = self.analyzer._calculate_bollinger_bands(df_current)
                        rsi = self.analyzer._calculate_rsi(df_current)
                        
                        technical_context = {
                            "macd": macd_data,
                            "rsi_5": rsi,
                            "bb_upper": bb_data['upper'].iloc[-1] if bb_data and 'upper' in bb_data else 0.0,
                            "bb_lower": bb_data['lower'].iloc[-1] if bb_data and 'lower' in bb_data else 0.0
                        }
                        
                        combo_result = self.combo_evaluator.evaluate_signal(
                            df=df_current,
                            direction_bias=direction,
                            technical_context=technical_context
                        )
                        
                        combo_score = combo_result['score']
                        combo_strength = combo_result['strength']
                        
                        # Apply combo bonus/penalty to confidence
                        if combo_strength == 'STRONG':
                            signal.confidence += 10
                        elif combo_strength == 'MEDIUM':
                            signal.confidence += 5
                        elif combo_strength == 'WEAK':
                            signal.confidence += 0
                        else:  # INVALID
                            signal.confidence -= 5
                    
                    # Only accept signals above minimum confidence
                    if signal.confidence >= MIN_SIGNAL_CONFIDENCE:
                        signal_data = {
                            'timestamp': df_current.index[-1],
                            'bar_index': i,
                            'pattern': pattern_name,
                            'direction': signal.debug_info.get('direction', 'LONG'),
                            'entry_price': signal.entry_price,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit,
                            'confidence': signal.confidence,
                            'risk_reward': signal.risk_reward_ratio,
                            'combo_score': combo_score,
                            'combo_strength': combo_strength,
                            'outcome': None,  # To be determined
                            'pnl': 0.0,
                            'exit_price': 0.0,
                            'bars_held': 0
                        }
                        
                        signals_found.append(signal_data)
                
            except Exception as e:
                # Skip bars with errors
                continue
        
        self.signals = signals_found
        logger.info(f"✅ Found {len(signals_found)} signals")
        return signals_found
    
    def _build_context(self, df: pd.DataFrame, sr: TechnicalLevels) -> Dict:
        """Build higher timeframe context for pattern detection."""
        try:
            current_price = df['close'].iloc[-1]
            rsi = self.analyzer._calculate_rsi(df)
            
            # Simple trend detection
            ema_20 = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
            trend = "UP" if current_price > ema_20 else "DOWN"
            
            return {
                "trend_direction": trend,
                "rsi_15": rsi,
                "price_above_vwap": True,
                "price_above_ema20": current_price > ema_20,
                "vwap_slope": trend,
                "prev_day_trend": trend,
                "vwap_5m": current_price * 0.998,
                "ema_20_5m": ema_20,
                "rsi_long_threshold": 60,
                "rsi_short_threshold": 40
            }
        except:
            return {
                "trend_direction": "FLAT",
                "rsi_15": 50,
                "price_above_vwap": True,
                "price_above_ema20": True,
                "vwap_slope": "FLAT",
                "prev_day_trend": "FLAT",
                "vwap_5m": 0,
                "ema_20_5m": 0
            }
    
    def evaluate_outcomes(self, df: pd.DataFrame):
        """Evaluate signal outcomes based on subsequent price action."""
        logger.info(f"📊 Evaluating {len(self.signals)} signal outcomes...")
        
        for signal in self.signals:
            bar_idx = signal['bar_index']
            entry = signal['entry_price']
            sl = signal['stop_loss']
            tp = signal['take_profit']
            direction = signal['direction']
            
            # Look at next 30 bars (max holding period ~2.5 hours)
            end_idx = min(bar_idx + 30, len(df) - 1)
            future_bars = df.iloc[bar_idx+1:end_idx+1]
            
            if len(future_bars) == 0:
                signal['outcome'] = 'OPEN'
                continue
            
            # Check for SL/TP hit
            hit_sl = False
            hit_tp = False
            exit_price = entry
            bars_held = len(future_bars)
            
            for idx, (timestamp, bar) in enumerate(future_bars.iterrows()):
                if direction == "LONG":
                    # Check SL
                    if bar['low'] <= sl:
                        hit_sl = True
                        exit_price = sl
                        bars_held = idx + 1
                        break
                    # Check TP
                    if bar['high'] >= tp:
                        hit_tp = True
                        exit_price = tp
                        bars_held = idx + 1
                        break
                else:  # SHORT
                    # Check SL
                    if bar['high'] >= sl:
                        hit_sl = True
                        exit_price = sl
                        bars_held = idx + 1
                        break
                    # Check TP
                    if bar['low'] <= tp:
                        hit_tp = True
                        exit_price = tp
                        bars_held = idx + 1
                        break
            
            # If no SL/TP hit, exit at last bar
            if not hit_sl and not hit_tp:
                exit_price = future_bars.iloc[-1]['close']
            
            # Calculate P&L
            if direction == "LONG":
                pnl_pct = ((exit_price - entry) / entry) * 100
            else:  # SHORT
                pnl_pct = ((entry - exit_price) / entry) * 100
            
            # Determine outcome
            if hit_tp:
                outcome = "WIN_TP"
            elif hit_sl:
                outcome = "LOSS_SL"
            elif pnl_pct > 0.5:  # Profitable close
                outcome = "WIN_CLOSE"
            elif pnl_pct < -0.5:  # Loss close
                outcome = "LOSS_CLOSE"
            else:
                outcome = "BREAKEVEN"
            
            signal['outcome'] = outcome
            signal['pnl'] = pnl_pct
            signal['exit_price'] = exit_price
            signal['bars_held'] = bars_held
        
        logger.info("✅ Outcome evaluation complete")
    
    def calculate_stats(self) -> Dict:
        """Calculate performance statistics."""
        if not self.signals:
            return {}
        
        total_signals = len(self.signals)
        wins = [s for s in self.signals if 'WIN' in s['outcome']]
        losses = [s for s in self.signals if 'LOSS' in s['outcome']]
        
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_signals * 100) if total_signals > 0 else 0
        
        avg_win = sum(s['pnl'] for s in wins) / len(wins) if wins else 0
        avg_loss = sum(s['pnl'] for s in losses) / len(losses) if losses else 0
        
        total_pnl = sum(s['pnl'] for s in self.signals)
        avg_pnl = total_pnl / total_signals if total_signals > 0 else 0
        
        avg_rr = sum(s['risk_reward'] for s in self.signals) / total_signals if total_signals > 0 else 0
        avg_confidence = sum(s['confidence'] for s in self.signals) / total_signals if total_signals > 0 else 0
        avg_bars_held = sum(s['bars_held'] for s in self.signals) / total_signals if total_signals > 0 else 0
        
        # Combo-specific stats
        if self.use_combo:
            strong_signals = [s for s in self.signals if s['combo_strength'] == 'STRONG']
            medium_signals = [s for s in self.signals if s['combo_strength'] == 'MEDIUM']
            weak_signals = [s for s in self.signals if s['combo_strength'] == 'WEAK']
            
            strong_win_rate = (len([s for s in strong_signals if 'WIN' in s['outcome']]) / len(strong_signals) * 100) if strong_signals else 0
            medium_win_rate = (len([s for s in medium_signals if 'WIN' in s['outcome']]) / len(medium_signals) * 100) if medium_signals else 0
            weak_win_rate = (len([s for s in weak_signals if 'WIN' in s['outcome']]) / len(weak_signals) * 100) if weak_signals else 0
        else:
            strong_signals = []
            medium_signals = []
            weak_signals = []
            strong_win_rate = 0
            medium_win_rate = 0
            weak_win_rate = 0
        
        return {
            'total_signals': total_signals,
            'wins': win_count,
            'losses': loss_count,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_rr': avg_rr,
            'avg_confidence': avg_confidence,
            'avg_bars_held': avg_bars_held,
            'strong_signals': len(strong_signals),
            'medium_signals': len(medium_signals),
            'weak_signals': len(weak_signals),
            'strong_win_rate': strong_win_rate,
            'medium_win_rate': medium_win_rate,
            'weak_win_rate': weak_win_rate
        }


def run_backtest_comparison():
    """Run backtest with and without combo strategy."""
    logger.info("="*80)
    logger.info("🚀 BACKTEST: COMBO STRATEGY vs. BASELINE")
    logger.info("="*80)
    logger.info("")
    
    # Fetch historical data (shared between both tests)
    logger.info("📥 Fetching historical data...")
    engine_baseline = BacktestEngine(use_combo=False)
    df = engine_baseline.fetch_data(days=14)
    
    if df is None or df.empty:
        logger.error("❌ Failed to fetch data")
        return
    
    logger.info(f"✅ Data ready: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    logger.info("")
    
    # Test 1: Without Combo
    logger.info("="*80)
    logger.info("TEST 1: BASELINE (Without Combo)")
    logger.info("="*80)
    
    baseline_signals = engine_baseline.detect_signals(df)
    engine_baseline.evaluate_outcomes(df)
    baseline_stats = engine_baseline.calculate_stats()
    
    # Test 2: With Combo
    logger.info("")
    logger.info("="*80)
    logger.info("TEST 2: WITH COMBO STRATEGY")
    logger.info("="*80)
    
    engine_combo = BacktestEngine(use_combo=True)
    combo_signals = engine_combo.detect_signals(df)
    engine_combo.evaluate_outcomes(df)
    combo_stats = engine_combo.calculate_stats()
    
    # Display Results
    logger.info("")
    logger.info("="*80)
    logger.info("📊 BACKTEST RESULTS COMPARISON")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("📈 PERFORMANCE COMPARISON")
    print("="*80)
    
    # Summary Table
    print(f"\n{'Metric':<25} {'Baseline':<20} {'With Combo':<20} {'Change':<15}")
    print("-" * 80)
    
    def format_change(baseline_val, combo_val, is_percentage=True, reverse=False):
        """Format the change with color indicators."""
        if baseline_val == 0:
            return "N/A"
        
        if is_percentage:
            change = combo_val - baseline_val
        else:
            change_pct = ((combo_val - baseline_val) / baseline_val) * 100
            change = change_pct
        
        # Reverse for metrics where lower is better
        if reverse:
            symbol = "📉" if change > 0 else ("📈" if change < 0 else "➖")
        else:
            symbol = "📈" if change > 0 else ("📉" if change < 0 else "➖")
        
        return f"{symbol} {change:+.1f}%"
    
    print(f"Total Signals         {baseline_stats['total_signals']:<20} {combo_stats['total_signals']:<20} {format_change(baseline_stats['total_signals'], combo_stats['total_signals'], False)}")
    print(f"Wins                  {baseline_stats['wins']:<20} {combo_stats['wins']:<20} {format_change(baseline_stats['wins'], combo_stats['wins'], False)}")
    print(f"Losses                {baseline_stats['losses']:<20} {combo_stats['losses']:<20} {format_change(baseline_stats['losses'], combo_stats['losses'], False, reverse=True)}")
    print(f"Win Rate              {baseline_stats['win_rate']:.1f}%{'':<15} {combo_stats['win_rate']:.1f}%{'':<15} {format_change(baseline_stats['win_rate'], combo_stats['win_rate'])}")
    print(f"Avg Win               {baseline_stats['avg_win']:.2f}%{'':<14} {combo_stats['avg_win']:.2f}%{'':<14} {format_change(baseline_stats['avg_win'], combo_stats['avg_win'])}")
    print(f"Avg Loss              {baseline_stats['avg_loss']:.2f}%{'':<14} {combo_stats['avg_loss']:.2f}%{'':<14} {format_change(baseline_stats['avg_loss'], combo_stats['avg_loss'], reverse=True)}")
    print(f"Total P&L             {baseline_stats['total_pnl']:.2f}%{'':<14} {combo_stats['total_pnl']:.2f}%{'':<14} {format_change(baseline_stats['total_pnl'], combo_stats['total_pnl'])}")
    print(f"Avg P&L/Trade         {baseline_stats['avg_pnl']:.2f}%{'':<14} {combo_stats['avg_pnl']:.2f}%{'':<14} {format_change(baseline_stats['avg_pnl'], combo_stats['avg_pnl'])}")
    print(f"Avg R:R               {baseline_stats['avg_rr']:.2f}x{'':<15} {combo_stats['avg_rr']:.2f}x{'':<15} {format_change(baseline_stats['avg_rr'], combo_stats['avg_rr'], False)}")
    print(f"Avg Confidence        {baseline_stats['avg_confidence']:.1f}%{'':<15} {combo_stats['avg_confidence']:.1f}%{'':<15} {format_change(baseline_stats['avg_confidence'], combo_stats['avg_confidence'])}")
    print(f"Avg Hold Time         {baseline_stats['avg_bars_held']:.1f} bars{'':<12} {combo_stats['avg_bars_held']:.1f} bars{'':<12} -")
    
    # Combo-specific breakdown
    if combo_stats['total_signals'] > 0:
        print("\n" + "="*80)
        print("🎯 COMBO STRENGTH BREAKDOWN (With Combo Only)")
        print("="*80)
        print(f"\n{'Strength':<15} {'Signals':<15} {'Win Rate':<15} {'Distribution':<15}")
        print("-" * 60)
        
        total = combo_stats['total_signals']
        print(f"STRONG          {combo_stats['strong_signals']:<15} {combo_stats['strong_win_rate']:.1f}%{'':<11} {combo_stats['strong_signals']/total*100:.1f}%")
        print(f"MEDIUM          {combo_stats['medium_signals']:<15} {combo_stats['medium_win_rate']:.1f}%{'':<11} {combo_stats['medium_signals']/total*100:.1f}%")
        print(f"WEAK            {combo_stats['weak_signals']:<15} {combo_stats['weak_win_rate']:.1f}%{'':<11} {combo_stats['weak_signals']/total*100:.1f}%")
    
    # Key Insights
    print("\n" + "="*80)
    print("💡 KEY INSIGHTS")
    print("="*80)
    
    insights = []
    
    # Win rate change
    wr_change = combo_stats['win_rate'] - baseline_stats['win_rate']
    if wr_change > 3:
        insights.append(f"✅ Win rate improved by {wr_change:.1f}% (combo filtering works!)")
    elif wr_change < -3:
        insights.append(f"⚠️ Win rate decreased by {abs(wr_change):.1f}% (may be over-filtering)")
    else:
        insights.append(f"➖ Win rate similar ({wr_change:+.1f}% change)")
    
    # Total P&L change
    pnl_change = combo_stats['total_pnl'] - baseline_stats['total_pnl']
    if pnl_change > baseline_stats['total_pnl'] * 0.1:
        insights.append(f"✅ Total P&L improved by {pnl_change:.2f}% (better profitability)")
    elif pnl_change < 0:
        insights.append(f"⚠️ Total P&L decreased by {abs(pnl_change):.2f}%")
    else:
        insights.append(f"➖ Total P&L similar ({pnl_change:+.2f}% change)")
    
    # Signal quality
    if combo_stats['total_signals'] < baseline_stats['total_signals']:
        reduction = (1 - combo_stats['total_signals']/baseline_stats['total_signals']) * 100
        insights.append(f"✅ Reduced signals by {reduction:.1f}% (better quality filtering)")
    
    # Combo effectiveness
    if combo_stats['strong_signals'] > 0 and combo_stats['strong_win_rate'] > combo_stats['win_rate']:
        insights.append(f"✅ STRONG signals have {combo_stats['strong_win_rate']:.1f}% win rate (combo working!)")
    
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Recommendation
    print("\n" + "="*80)
    print("🎯 RECOMMENDATION")
    print("="*80)
    
    if combo_stats['win_rate'] > baseline_stats['win_rate'] and combo_stats['total_pnl'] > baseline_stats['total_pnl']:
        print("✅ ENABLE COMBO STRATEGY - Improves both win rate and profitability")
    elif combo_stats['win_rate'] > baseline_stats['win_rate']:
        print("⚠️ CONSIDER COMBO - Improves win rate but check profitability")
    elif combo_stats['total_pnl'] > baseline_stats['total_pnl']:
        print("⚠️ CONSIDER COMBO - Improves profitability but lower win rate")
    else:
        print("⚠️ KEEP BASELINE - Combo didn't improve performance on this data")
    
    print("\n" + "="*80)
    print("✅ BACKTEST COMPLETE")
    print("="*80)
    
    return baseline_stats, combo_stats


if __name__ == "__main__":
    run_backtest_comparison()
