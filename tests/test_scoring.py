
import unittest
import logging
from unittest.mock import MagicMock
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import NiftyTradingAgent
from analysis_module.technical import Signal, SignalType
from analysis_module.option_chain_analyzer import OptionChainAnalyzer

# Logging
logging.basicConfig(level=logging.INFO)

class TestWeightedScoring(unittest.TestCase):
    def setUp(self):
        self.agent = NiftyTradingAgent()
        self.agent.option_fetcher = MagicMock()
        self.agent.option_analyzer = OptionChainAnalyzer()
        self.agent.fetcher = MagicMock()
        self.agent.fetcher.get_historical_data.return_value = None # For choppy check (mock it out)
        
        # Patch TechnicalAnalyzer inside main.py if needed, 
        # but _generate_signals creates a new instance.
        # We can mock `_is_choppy_session` on the class or just ensure it returns False.
        # Ideally, we mock TechnicalAnalyzer entirely.
        
    @unittest.mock.patch('main.TechnicalAnalyzer')
    def test_scoring_logic_bullish(self, MockAnalyzer):
        """Test Scoring System for a Bullish Signal"""
        # 1. Setup Mock Analyzer to avoid Choppy Session
        mock_instance = MockAnalyzer.return_value
        mock_instance._is_choppy_session.return_value = (False, "Trend")
        
        # 2. Setup Option Chain Data (Bullish PCR)
        # PCR = 200/100 = 2.0 (Deep Bullish)
        mock_oc_data = {
            "records": {"data": [
                 {"strikePrice": 26100, "CE": {"openInterest": 100}, "PE": {"openInterest": 200}}
            ]}
        }
        self.agent.option_fetcher.fetch_option_chain.return_value = mock_oc_data
        
        # 3. Create a Signal (Weak Technicals but Strong Options)
        # Conf 65 (+10 pts)
        # Volume Confirmed (+10 pts)
        # Base = 50
        # Total from Technicals = 50 + 10 + 10 = 70
        # Options: PCR Bullish on Bullish Signal -> +10
        # Total Score = 80
        
        sig = Signal(
            signal_type=SignalType.BULLISH_BREAKOUT,
            instrument="NIFTY",
            timeframe="5m",
            price_level=26100.0,
            entry_price=26110.0,
            stop_loss=26080.0,
            take_profit=26200.0,
            confidence=65.0,
            timestamp=datetime.now(),
            description="Breakout",
            risk_reward_ratio=3.0,
            volume_confirmed=True,
            momentum_confirmed=True
        )
        
        analysis_input = {"breakout_signal": sig}
        nse_data_input = {"lastPrice": 26105, "price": 26105}
        
        # 4. Generate Signals
        signals = self.agent._generate_signals("NIFTY", analysis_input, nse_data_input)
        
        # 5. Verify Score
        self.assertTrue(len(signals) > 0, "Signal should be accepted")
        result_sig = signals[0]
        score = result_sig.get("score", 0)
        reasons = result_sig.get("score_reasons", [])
        
        print(f"\n✅ Result Score: {score}")
        print(f"📝 Reasons: {reasons}")
        
        # Assertions
        self.assertGreaterEqual(score, 70)
        self.assertIn("PCR Bullish", str(reasons))
        self.assertIn("Volume High (+10)", reasons)

    @unittest.mock.patch('main.TechnicalAnalyzer')
    def test_scoring_rejection(self, MockAnalyzer):
        """Test Rejection of Weak Signal"""
        # 1. Setup Mock
        mock_instance = MockAnalyzer.return_value
        mock_instance._is_choppy_session.return_value = (False, "Trend")
        
        # 2. Setup Bearish Options for a Bullish Signal
        # PCR = 0.5 (Bearish)
        mock_oc_data = {
            "records": {"data": [
                 {"strikePrice": 26100, "CE": {"openInterest": 200}, "PE": {"openInterest": 100}}
            ]}
        }
        self.agent.option_fetcher.fetch_option_chain.return_value = mock_oc_data
        
        # 3. Weak Bullish Signal
        # Base = 50
        # Conf 55 (<65) -> +0 pts
        # Volume False -> +0 pts
        # Options: PCR Bearish (0.5) on Bullish Signal -> -10 pts
        # Total Score = 40
        # Threshold 60 -> REJECT
        
        sig = Signal(
            signal_type=SignalType.BULLISH_BREAKOUT,
            instrument="NIFTY",
            timeframe="5m",
            price_level=26100.0,
            entry_price=26110.0,
            stop_loss=26080.0,
            take_profit=26200.0,
            confidence=55.0,
            timestamp=datetime.now(),
            description="Weak Breakout",
            risk_reward_ratio=3.0,
            volume_confirmed=False,
            momentum_confirmed=True
        )
        
        analysis_input = {"breakout_signal": sig}
        nse_data_input = {"lastPrice": 26105, "price": 26105}
        
        # 4. Generate
        signals = self.agent._generate_signals("NIFTY", analysis_input, nse_data_input)
        
        # 5. Verify Rejection
        self.assertEqual(len(signals), 0, "Weak signal should be rejected by scoring")
        print("\n✅ Weak Signal Rejected as expected.")

    @unittest.mock.patch('main.TechnicalAnalyzer')
    def test_mtf_alignment_boost(self, MockAnalyzer):
        """Test MTF Trend Alignment Boost (+15)"""
        # 1. Setup Mock (Aligned Trend)
        mock_instance = MockAnalyzer.return_value
        mock_instance._is_choppy_session.return_value = (False, "Trend")
        
        # 2. Options (Neutral)
        mock_oc_data = {
            "records": {"data": [
                 {"strikePrice": 26100, "CE": {"openInterest": 100}, "PE": {"openInterest": 100}}
            ]}
        }
        self.agent.option_fetcher.fetch_option_chain.return_value = mock_oc_data
        
        # 3. Moderate Signal
        # Base = 50
        # Conf 70 (+10)
        # Volume False
        # MTF Trend UP (Bullish) -> +15
        # Total = 75
        
        sig = Signal(
            signal_type=SignalType.BULLISH_BREAKOUT,
            instrument="NIFTY",
            timeframe="5m",
            price_level=26100.0,
            entry_price=26110.0,
            stop_loss=26080.0,
            take_profit=26200.0,
            confidence=70.0,
            timestamp=datetime.now(),
            description="Breakout",
            risk_reward_ratio=3.0,
            volume_confirmed=False,
            momentum_confirmed=True
        )
        
        # Inject Context logic via main.py modification assumption
        # Note: We need to verify that main.py logic picks this up.
        # Since we can't easily inject into the internal variable of _generate_signals without
        # using integration test style or modifying how we pass analysis,
        # we rely on the fact that _generate_signals receives 'analysis' dict.
        
        analysis_input = {
            "breakout_signal": sig,
            "higher_tf_context": {"trend_direction": "UP"}
        }
        nse_data_input = {"lastPrice": 26105, "price": 26105}
        
        signals = self.agent._generate_signals("NIFTY", analysis_input, nse_data_input)
        
        self.assertTrue(len(signals) > 0)
        self.assertGreaterEqual(signals[0]["score"], 75)
        self.assertIn("Trend Aligned", str(signals[0]["score_reasons"]))
        print(f"\n✅ MTF Alignment Boost Verified: Score {signals[0]['score']}")

    @unittest.mock.patch('main.TechnicalAnalyzer')
    def test_mtf_conflict_penalty(self, MockAnalyzer):
        """Test MTF Trend Conflict Penalty (-15)"""
        # 1. Setup Mock
        mock_instance = MockAnalyzer.return_value
        mock_instance._is_choppy_session.return_value = (False, "Trend")
        
        # 2. Options (Neutral)
        # 3. Good Signal but Counter Trend
        # Base = 50
        # Conf 70 (+10)
        # Volume True (+10)
        # MTF Trend DOWN (Bearish vs Bullish Sig) -> -15
        # Total = 70 - 15 = 55 (REJECT < 60)
        
        mock_oc_data = {"records": {"data": [{"strikePrice": 26100, "CE": {"openInterest": 100}, "PE": {"openInterest": 100}}]}}
        self.agent.option_fetcher.fetch_option_chain.return_value = mock_oc_data
        
        sig = Signal(
            signal_type=SignalType.BULLISH_BREAKOUT,
            instrument="NIFTY",
            timeframe="5m",
            price_level=26100.0,
            entry_price=26110.0,
            stop_loss=26080.0,
            take_profit=26200.0,
            confidence=70.0,
            timestamp=datetime.now(),
            description="Counter Trend Breakout",
            risk_reward_ratio=3.0,
            volume_confirmed=True,
            momentum_confirmed=True
        )
        
        analysis_input = {
            "breakout_signal": sig,
            "higher_tf_context": {"trend_direction": "DOWN"}
        }
        nse_data_input = {"lastPrice": 26105, "price": 26105}
        
        signals = self.agent._generate_signals("NIFTY", analysis_input, nse_data_input)
        
        self.assertEqual(len(signals), 0, "Counter-trend signal should be rejected (Score ~55)")
        print("\n✅ MTF Conflict Penalty Verified: Signal Rejected")

if __name__ == '__main__':
    unittest.main()
