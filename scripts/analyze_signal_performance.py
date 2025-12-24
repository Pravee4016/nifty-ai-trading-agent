#!/usr/bin/env python3
"""
Signal Performance Analyzer
Analyzes trading signals against actual market data
"""

import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yfinance as yf
import pandas as pd
from dataclasses import dataclass
import pytz


@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    direction: str
    signal_type: str
    entry: float
    stop_loss: float
    target: float
    rr_ratio: float
    confidence: float
    score: int
    ai_verdict: str
    ai_confidence: int
    
    def __str__(self):
        return (f"{self.timestamp.strftime('%H:%M')} | {self.symbol} | {self.direction} | "
                f"Entry: {self.entry} | SL: {self.stop_loss} | Target: {self.target} | "
                f"RR: {self.rr_ratio} | Conf: {self.confidence}%")


class SignalPerformanceAnalyzer:
    def __init__(self):
        self.signals: List[Signal] = []
        
    def parse_signals(self, signal_text: str):
        """Parse signals from Telegram message format"""
        # Split by signal blocks
        blocks = signal_text.split('\n\n')
        
        current_signal = {}
        for block in blocks:
            lines = block.strip().split('\n')
            
            for line in lines:
                # Symbol
                if '📊' in line:
                    current_signal['symbol'] = line.split('📊')[1].strip()
                
                # Entry
                if '💰 Entry:' in line:
                    current_signal['entry'] = float(re.search(r'(\d+\.?\d*)', line).group(1))
                
                # Stop Loss
                if '🛑' in line and ('SL:' in line or 'Stop Loss:' in line):
                    current_signal['stop_loss'] = float(re.search(r'(\d+\.?\d*)', line).group(1))
                
                # Target
                if '🎯 Target' in line:
                    current_signal['target'] = float(re.search(r'(\d+\.?\d*)', line).group(1))
                
                # Risk:Reward
                if 'Risk:Reward:' in line or 'RR:' in line:
                    rr_match = re.search(r'(\d+\.?\d*):(\d+\.?\d*)', line)
                    if rr_match:
                        current_signal['rr_ratio'] = float(rr_match.group(2))
                
                # Confidence
                if '⚡ Confidence:' in line:
                    current_signal['confidence'] = float(re.search(r'(\d+\.?\d*)%', line).group(1))
                
                # Score
                if '🏆 Score:' in line:
                    current_signal['score'] = int(re.search(r'(\d+)/100', line).group(1))
                
                # Timestamp
                if '⏰' in line:
                    time_str = line.split('⏰')[1].strip()
                    # Parse as IST and convert to UTC for comparison with market data
                    ist = pytz.timezone('Asia/Kolkata')
                    naive_dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S IST')
                    ist_dt = ist.localize(naive_dt)
                    current_signal['timestamp'] = ist_dt.astimezone(pytz.UTC)
                
                # Signal type and direction
                if '🚀 BREAKOUT' in line or '🎯 RETEST SETUP' in line:
                    if 'BREAKOUT' in line:
                        current_signal['signal_type'] = 'BREAKOUT'
                    else:
                        current_signal['signal_type'] = 'RETEST'
                    
                    if '📈 LONG' in block or '💡 Bullish' in block or '💡 Support retest' in block:
                        current_signal['direction'] = 'LONG'
                    else:
                        current_signal['direction'] = 'SHORT'
                
                # AI Verdict
                if '• Verdict:' in line:
                    verdict_match = re.search(r'Verdict: (\w+).*?\((\d+)%\)', line)
                    if verdict_match:
                        current_signal['ai_verdict'] = verdict_match.group(1)
                        current_signal['ai_confidence'] = int(verdict_match.group(2))
            
            # Create signal if we have minimum required fields
            if all(k in current_signal for k in ['symbol', 'entry', 'stop_loss', 'target']):
                # Ensure timestamp is UTC-aware
                timestamp = current_signal.get('timestamp')
                if timestamp is None:
                    timestamp = datetime.now(pytz.UTC)
                elif timestamp.tzinfo is None:
                    # Should not happen with our parsing, but just in case
                    timestamp = pytz.UTC.localize(timestamp)
                
                signal = Signal(
                    timestamp=timestamp,
                    symbol=current_signal['symbol'],
                    direction=current_signal.get('direction', 'UNKNOWN'),
                    signal_type=current_signal.get('signal_type', 'UNKNOWN'),
                    entry=current_signal['entry'],
                    stop_loss=current_signal['stop_loss'],
                    target=current_signal['target'],
                    rr_ratio=current_signal.get('rr_ratio', 0),
                    confidence=current_signal.get('confidence', 0),
                    score=current_signal.get('score', 0),
                    ai_verdict=current_signal.get('ai_verdict', 'UNKNOWN'),
                    ai_confidence=current_signal.get('ai_confidence', 0)
                )
                self.signals.append(signal)
                current_signal = {}
    
    def fetch_market_data(self, symbol: str, date: datetime) -> Optional[pd.DataFrame]:
        """Fetch intraday market data for analysis"""
        try:
            # Map symbols to Yahoo Finance tickers
            ticker_map = {
                'NIFTY': '^NSEI',
                'BANKNIFTY': '^NSEBANK'
            }
            ticker = ticker_map.get(symbol, symbol)
            
            # Fetch 1-minute data for the day
            start = date.replace(hour=9, minute=0, second=0)
            end = date.replace(hour=15, minute=30, second=0)
            
            data = yf.download(ticker, start=start, end=end, interval='1m', progress=False)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def analyze_signal(self, signal: Signal, market_data: pd.DataFrame) -> Dict:
        """Analyze if a signal hit target or stop loss"""
        result = {
            'signal': signal,
            'outcome': 'UNKNOWN',
            'hit_time': None,
            'hit_price': None,
            'points_achieved': 0,
            'rr_achieved': 0,
            'duration_minutes': 0
        }
        
        if market_data is None or market_data.empty:
            return result
        
        # Get data after signal time
        after_signal = market_data[market_data.index >= signal.timestamp]
        
        if after_signal.empty:
            return result
        
        if signal.direction == 'LONG':
            # Check if target hit first
            target_hit = after_signal[after_signal['High'] >= signal.target]
            sl_hit = after_signal[after_signal['Low'] <= signal.stop_loss]
            
            if not target_hit.empty and not sl_hit.empty:
                # Both hit, check which came first
                if target_hit.index[0] <= sl_hit.index[0]:
                    result['outcome'] = 'TARGET'
                    result['hit_time'] = target_hit.index[0]
                    result['hit_price'] = signal.target
                    result['points_achieved'] = signal.target - signal.entry
                else:
                    result['outcome'] = 'STOPLOSS'
                    result['hit_time'] = sl_hit.index[0]
                    result['hit_price'] = signal.stop_loss
                    result['points_achieved'] = signal.stop_loss - signal.entry
            elif not target_hit.empty:
                result['outcome'] = 'TARGET'
                result['hit_time'] = target_hit.index[0]
                result['hit_price'] = signal.target
                result['points_achieved'] = signal.target - signal.entry
            elif not sl_hit.empty:
                result['outcome'] = 'STOPLOSS'
                result['hit_time'] = sl_hit.index[0]
                result['hit_price'] = signal.stop_loss
                result['points_achieved'] = signal.stop_loss - signal.entry
            else:
                result['outcome'] = 'OPEN'
                # Use last available price
                last_price = after_signal.iloc[-1]['Close']
                result['hit_price'] = last_price
                result['points_achieved'] = last_price - signal.entry
        
        else:  # SHORT
            # Check if target hit first
            target_hit = after_signal[after_signal['Low'] <= signal.target]
            sl_hit = after_signal[after_signal['High'] >= signal.stop_loss]
            
            if not target_hit.empty and not sl_hit.empty:
                # Both hit, check which came first
                if target_hit.index[0] <= sl_hit.index[0]:
                    result['outcome'] = 'TARGET'
                    result['hit_time'] = target_hit.index[0]
                    result['hit_price'] = signal.target
                    result['points_achieved'] = signal.entry - signal.target
                else:
                    result['outcome'] = 'STOPLOSS'
                    result['hit_time'] = sl_hit.index[0]
                    result['hit_price'] = signal.stop_loss
                    result['points_achieved'] = signal.entry - signal.stop_loss
            elif not target_hit.empty:
                result['outcome'] = 'TARGET'
                result['hit_time'] = target_hit.index[0]
                result['hit_price'] = signal.target
                result['points_achieved'] = signal.entry - signal.target
            elif not sl_hit.empty:
                result['outcome'] = 'STOPLOSS'
                result['hit_time'] = sl_hit.index[0]
                result['hit_price'] = signal.stop_loss
                result['points_achieved'] = signal.entry - signal.stop_loss
            else:
                result['outcome'] = 'OPEN'
                # Use last available price
                last_price = after_signal.iloc[-1]['Close']
                result['hit_price'] = last_price
                result['points_achieved'] = signal.entry - last_price
        
        # Calculate RR achieved
        risk = abs(signal.entry - signal.stop_loss)
        if risk > 0:
            result['rr_achieved'] = result['points_achieved'] / risk
        
        # Calculate duration
        if result['hit_time']:
            duration = (result['hit_time'] - signal.timestamp).total_seconds() / 60
            result['duration_minutes'] = int(duration)
        
        return result
    
    def generate_report(self, results: List[Dict], symbol: str) -> str:
        """Generate performance report for a symbol"""
        if not results:
            return f"\n{'='*80}\n{symbol} - No signals found\n{'='*80}\n"
        
        report = []
        report.append(f"\n{'='*80}")
        report.append(f"{symbol} SIGNAL PERFORMANCE ANALYSIS - {datetime.now().strftime('%Y-%m-%d')}")
        report.append(f"{'='*80}\n")
        
        # Overall statistics
        total = len(results)
        wins = sum(1 for r in results if r['outcome'] == 'TARGET')
        losses = sum(1 for r in results if r['outcome'] == 'STOPLOSS')
        open_trades = sum(1 for r in results if r['outcome'] == 'OPEN')
        unknown = sum(1 for r in results if r['outcome'] == 'UNKNOWN')
        
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        
        total_points = sum(r['points_achieved'] for r in results if r['outcome'] in ['TARGET', 'STOPLOSS'])
        avg_points = total_points / (wins + losses) if (wins + losses) > 0 else 0
        
        avg_win_points = sum(r['points_achieved'] for r in results if r['outcome'] == 'TARGET') / wins if wins > 0 else 0
        avg_loss_points = sum(r['points_achieved'] for r in results if r['outcome'] == 'STOPLOSS') / losses if losses > 0 else 0
        
        report.append("📊 OVERALL STATISTICS")
        report.append(f"Total Signals: {total}")
        report.append(f"Wins: {wins} | Losses: {losses} | Open: {open_trades} | Unknown: {unknown}")
        report.append(f"Win Rate: {win_rate:.1f}%")
        report.append(f"Total Points: {total_points:.2f}")
        report.append(f"Average Points/Trade: {avg_points:.2f}")
        report.append(f"Average Win: {avg_win_points:.2f} pts | Average Loss: {avg_loss_points:.2f} pts")
        
        if wins > 0 and losses > 0:
            profit_factor = abs(avg_win_points * wins / (avg_loss_points * losses))
            report.append(f"Profit Factor: {profit_factor:.2f}")
        
        report.append("")
        
        # Signal breakdown
        report.append("📝 SIGNAL-BY-SIGNAL BREAKDOWN")
        report.append("-" * 80)
        
        for i, result in enumerate(results, 1):
            signal = result['signal']
            outcome_emoji = {
                'TARGET': '✅',
                'STOPLOSS': '❌',
                'OPEN': '⏳',
                'UNKNOWN': '❓'
            }
            
            report.append(f"\n#{i} | {outcome_emoji.get(result['outcome'], '?')} {result['outcome']}")
            report.append(f"Time: {signal.timestamp.strftime('%H:%M:%S')}")
            report.append(f"Type: {signal.signal_type} {signal.direction}")
            report.append(f"Entry: {signal.entry} | SL: {signal.stop_loss} | Target: {signal.target}")
            report.append(f"Expected RR: {signal.rr_ratio} | Confidence: {signal.confidence}% | Score: {signal.score}/100")
            report.append(f"AI Verdict: {signal.ai_verdict} ({signal.ai_confidence}%)")
            
            if result['outcome'] in ['TARGET', 'STOPLOSS', 'OPEN']:
                report.append(f"Hit Price: {result['hit_price']:.2f} | Points: {result['points_achieved']:+.2f} | RR Achieved: {result['rr_achieved']:.2f}")
                if result['hit_time']:
                    report.append(f"Duration: {result['duration_minutes']} minutes")
            
            report.append("-" * 80)
        
        return '\n'.join(report)


def main():
    """Main analysis function"""
    # Signal data from today (Dec 19, 2025)
    signals_text = """
NiftyLevelsAI, [Dec 19, 2025 at 9:31 AM]
📉 🚀 BREAKOUT

📊 BANKNIFTY
💰 Entry: 59086.00
🛑 SL: 59255.29
🎯 Target 1: 58832.00 (Safe)
📈 RR: 1.50:1
⚡ Confidence: 85.0%
🏆 Score: 100/100

⏰ 2025-12-19 09:31:55 IST

🤖 AI Analyst Review (🔮 VERTEX)
• Verdict: CAUTIOUS_SELL (65%)

NiftyLevelsAI, [Dec 19, 2025 at 9:31 AM]
🔄 🎯 RETEST SETUP 📉 SHORT

📊 BANKNIFTY
💰 Entry: 59086.00
🛑 Stop Loss: 59173.30
🎯 Target 1: 58824.11 (Safe)
📈 Risk:Reward: 1:3.0
⚡ Confidence: 80%
🏆 Score: 100/100

⏰ 2025-12-19 09:31:58 IST

🤖 AI Analyst Review (🧠 GROQ)
• Verdict: CAUTIOUS_SELL (70%)

NiftyLevelsAI, [Dec 19, 2025 at 9:52 AM]
🔄 🎯 RETEST SETUP 📉 SHORT

📊 NIFTY
💰 Entry: 25940.80
🛑 Stop Loss: 25968.64
🎯 Target 1: 25857.27 (Safe)
📈 Risk:Reward: 1:3.0
⚡ Confidence: 75%
🏆 Score: 90/100

⏰ 2025-12-19 09:52:31 IST

🤖 AI Analyst Review (🧠 GROQ)
• Verdict: CAUTIOUS_SELL (65%)

NiftyLevelsAI, [Dec 19, 2025 at 10:01 AM]
🔄 🎯 RETEST SETUP 📉 SHORT

📊 NIFTY
💰 Entry: 25956.20
🛑 Stop Loss: 25982.26
🎯 Target 1: 25878.03 (Safe)
📈 Risk:Reward: 1:3.0
⚡ Confidence: 80%
🏆 Score: 85/100

⏰ 2025-12-19 10:01:45 IST

🤖 AI Analyst Review (🧠 GROQ)
• Verdict: CAUTIOUS_SELL (60%)

NiftyLevelsAI, [Dec 19, 2025 at 10:42 AM]
🎯 🎯 RETEST SETUP 📈 LONG

📊 BANKNIFTY
💰 Entry: 59099.70
🛑 Stop Loss: 58958.92
🎯 Target 1: 59311.00 (Safe)
📈 Risk:Reward: 1:1.5
⚡ Confidence: 75%
🏆 Score: 100/100

⏰ 2025-12-19 10:42:19 IST

🤖 AI Analyst Review (🔮 VERTEX)
• Verdict: CAUTIOUS_BUY (65%)

NiftyLevelsAI, [Dec 19, 2025 at 10:46 AM]
🔄 🎯 RETEST SETUP 📈 LONG

📊 BANKNIFTY
💰 Entry: 59100.50
🛑 Stop Loss: 59044.21
🎯 Target 1: 59269.37 (Safe)
📈 Risk:Reward: 1:3.0
⚡ Confidence: 80%
🏆 Score: 100/100

⏰ 2025-12-19 10:46:50 IST

🤖 AI Analyst Review (🔮 VERTEX)
• Verdict: CAUTIOUS_BUY (65%)

NiftyLevelsAI, [Dec 19, 2025 at 11:16 AM]
🎯 🎯 RETEST SETUP 📈 LONG

📊 BANKNIFTY
💰 Entry: 59014.15
🛑 Stop Loss: 58822.84
🎯 Target 1: 59301.00 (Safe)
📈 Risk:Reward: 1:1.5
⚡ Confidence: 75%
🏆 Score: 100/100

⏰ 2025-12-19 11:16:58 IST

🤖 AI Analyst Review (🔮 VERTEX)
• Verdict: CAUTIOUS_BUY (65%)

NiftyLevelsAI, [Dec 19, 2025 at 11:41 AM]
🔄 🎯 RETEST SETUP 📉 SHORT

📊 NIFTY
💰 Entry: 25932.30
🛑 Stop Loss: 26007.53
🎯 Target 1: 25819.00 (Safe)
📈 Risk:Reward: 1:1.5
⚡ Confidence: 65%
🏆 Score: 80/100

⏰ 2025-12-19 11:41:42 IST

🤖 AI Analyst Review (🔮 VERTEX)
• Verdict: CAUTIOUS_SELL (65%)

NiftyLevelsAI, [Dec 19, 2025 at 11:47 AM]
🔄 🎯 RETEST SETUP 📉 SHORT

📊 BANKNIFTY
💰 Entry: 58901.40
🛑 Stop Loss: 59045.82
🎯 Target 1: 58685.00 (Safe)
📈 Risk:Reward: 1:1.5
⚡ Confidence: 80%
🏆 Score: 100/100

⏰ 2025-12-19 11:47:31 IST

🤖 AI Analyst Review (🔮 VERTEX)
• Verdict: CAUTIOUS_SELL (65%)

NiftyLevelsAI, [Dec 19, 2025 at 12:16 PM]
🔄 🎯 RETEST SETUP 📉 SHORT

📊 NIFTY
💰 Entry: 25944.40
🛑 Stop Loss: 25967.37
🎯 Target 1: 25875.49 (Safe)
📈 Risk:Reward: 1:3.0
⚡ Confidence: 80%
🏆 Score: 95/100

⏰ 2025-12-19 12:16:58 IST

🤖 AI Analyst Review (🧠 GROQ)
• Verdict: CAUTIOUS_SELL (60%)

NiftyLevelsAI, [Dec 19, 2025 at 12:17 PM]
🎯 🎯 RETEST SETUP 📈 LONG

📊 BANKNIFTY
💰 Entry: 59014.45
🛑 Stop Loss: 58827.73
🎯 Target 1: 59295.00 (Safe)
📈 Risk:Reward: 1:1.5
⚡ Confidence: 75%
🏆 Score: 100/100

⏰ 2025-12-19 12:17:04 IST

🤖 AI Analyst Review (🔮 VERTEX)
• Verdict: CAUTIOUS_BUY (65%)

NiftyLevelsAI, [Dec 19, 2025 at 1:11 PM]
📉 🚀 BREAKOUT

📊 BANKNIFTY
💰 Entry: 58965.60
🛑 SL: 59095.75
🎯 Target 1: 58770.00 (Safe)
📈 RR: 1.50:1
⚡ Confidence: 90.0%
🏆 Score: 100/100

⏰ 2025-12-19 13:11:58 IST

🤖 AI Analyst Review (🧠 GROQ)
• Verdict: STRONG_SELL (85%)

NiftyLevelsAI, [Dec 19, 2025 at 1:32 PM]
🎯 🎯 RETEST SETUP 📈 LONG

📊 BANKNIFTY
💰 Entry: 59040.75
🛑 Stop Loss: 58843.35
🎯 Target 1: 59337.00 (Safe)
📈 Risk:Reward: 1:1.5
⚡ Confidence: 65%
🏆 Score: 100/100

⏰ 2025-12-19 13:32:11 IST

🤖 AI Analyst Review (🧠 GROQ)
• Verdict: CAUTIOUS_BUY (65%)

NiftyLevelsAI, [Dec 19, 2025 at 1:46 PM]
🎯 🎯 RETEST SETUP 📈 LONG

📊 BANKNIFTY
💰 Entry: 59104.30
🛑 Stop Loss: 58992.82
🎯 Target 1: 59272.00 (Safe)
📈 Risk:Reward: 1:1.5
⚡ Confidence: 75%
🏆 Score: 100/100

⏰ 2025-12-19 13:46:30 IST

🤖 AI Analyst Review (🧠 GROQ)
• Verdict: CAUTIOUS_BUY (70%)

NiftyLevelsAI, [Dec 19, 2025 at 1:52 PM]
🔄 🎯 RETEST SETUP 📈 LONG

📊 BANKNIFTY
💰 Entry: 59114.10
🛑 Stop Loss: 59069.82
🎯 Target 1: 59246.95 (Safe)
📈 Risk:Reward: 1:3.0
⚡ Confidence: 75%
🏆 Score: 100/100

⏰ 2025-12-19 13:52:34 IST

🤖 AI Analyst Review (🔮 VERTEX)
• Verdict: CAUTIOUS_BUY (65%)

NiftyLevelsAI, [Dec 19, 2025 at 2:26 PM]
🔄 🎯 RETEST SETUP 📉 SHORT

📊 NIFTY
💰 Entry: 25982.50
🛑 Stop Loss: 26054.03
🎯 Target 1: 25875.00 (Safe)
📈 Risk:Reward: 1:1.5
⚡ Confidence: 65%
🏆 Score: 80/100

⏰ 2025-12-19 14:26:31 IST

🤖 AI Analyst Review (🔮 VERTEX)
• Verdict: CAUTIOUS_SELL (65%)

NiftyLevelsAI, [Dec 19, 2025 at 2:56 PM]
🎯 🎯 RETEST SETUP 📈 LONG

📊 BANKNIFTY
💰 Entry: 59098.45
🛑 Stop Loss: 59000.71
🎯 Target 1: 59245.00 (Safe)
📈 Risk:Reward: 1:1.5
⚡ Confidence: 80%
🏆 Score: 100/100

⏰ 2025-12-19 14:56:56 IST

🤖 AI Analyst Review (🧠 GROQ)
• Verdict: CAUTIOUS_BUY (70%)
"""
    
    analyzer = SignalPerformanceAnalyzer()
    analyzer.parse_signals(signals_text)
    
    print(f"\n✅ Parsed {len(analyzer.signals)} signals")
    
    # Separate by symbol
    nifty_signals = [s for s in analyzer.signals if s.symbol == 'NIFTY']
    banknifty_signals = [s for s in analyzer.signals if s.symbol == 'BANKNIFTY']
    
    print(f"   - NIFTY: {len(nifty_signals)} signals")
    print(f"   - BANKNIFTY: {len(banknifty_signals)} signals")
    
    # Fetch market data
    today = datetime(2025, 12, 19)
    print(f"\n📊 Fetching market data for {today.strftime('%Y-%m-%d')}...")
    
    nifty_data = analyzer.fetch_market_data('NIFTY', today)
    banknifty_data = analyzer.fetch_market_data('BANKNIFTY', today)
    
    # Analyze signals
    print("\n🔍 Analyzing signals against market data...")
    
    nifty_results = []
    for signal in nifty_signals:
        result = analyzer.analyze_signal(signal, nifty_data)
        nifty_results.append(result)
    
    banknifty_results = []
    for signal in banknifty_signals:
        result = analyzer.analyze_signal(signal, banknifty_data)
        banknifty_results.append(result)
    
    # Generate reports
    nifty_report = analyzer.generate_report(nifty_results, 'NIFTY')
    banknifty_report = analyzer.generate_report(banknifty_results, 'BANKNIFTY')
    
    print(nifty_report)
    print(banknifty_report)
    
    # Save to file
    output_file = f"/Users/praveent/nifty-ai-trading-agent/signal_performance_{today.strftime('%Y%m%d')}.txt"
    with open(output_file, 'w') as f:
        f.write(nifty_report)
        f.write("\n\n")
        f.write(banknifty_report)
    
    print(f"\n💾 Report saved to: {output_file}")


if __name__ == "__main__":
    main()
