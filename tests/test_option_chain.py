
import unittest
import logging
from unittest.mock import MagicMock, patch
from analysis_module.option_chain_analyzer import OptionChainAnalyzer
from app.agent import NiftyTradingAgent

# Configure logging to show info during tests
logging.basicConfig(level=logging.INFO)

class TestOptionChainAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = OptionChainAnalyzer()
        # Mock Option Chain Data (Simplified structure)
        self.mock_data = {
            "records": {
                "data": [
                    {
                        "strikePrice": 26000,
                        "CE": {"openInterest": 100000, "changeinOpenInterest": 5000, "impliedVolatility": 12.0},
                        "PE": {"openInterest": 50000, "changeinOpenInterest": -2000, "impliedVolatility": 14.0}
                    },
                    {
                        "strikePrice": 26100, # ATM
                        "CE": {"openInterest": 80000, "changeinOpenInterest": 10000, "impliedVolatility": 11.5},
                        "PE": {"openInterest": 90000, "changeinOpenInterest": 15000, "impliedVolatility": 13.5}
                    },
                    {
                        "strikePrice": 26200,
                        "CE": {"openInterest": 60000, "changeinOpenInterest": -1000, "impliedVolatility": 11.0},
                        "PE": {"openInterest": 120000, "changeinOpenInterest": 20000, "impliedVolatility": 12.0}
                    }
                ]
            }
        }

    def test_pcr_calculation(self):
        """Test Put-Call Ratio Calculation"""
        # Total CE OI = 100k + 80k + 60k = 240k
        # Total PE OI = 50k + 90k + 120k = 260k
        # PCR = 260/240 = 1.0833
        pcr = self.analyzer.calculate_pcr(self.mock_data)
        self.assertAlmostEqual(pcr, 1.0833, places=2)
        print(f"\n✅ PCR Test Passed: Calculated {pcr}")

    def test_max_pain(self):
        """Test Max Pain Calculation"""
        # Max Pain logic is complex, verification of result:
        # At 26000: PE writers lose on nothing (ITM Puts only above). Calls below are ITM.
        # This is a functional test to ensure it returns a strike from the list.
        mp = self.analyzer.calculate_max_pain(self.mock_data)
        self.assertIn(mp, [26000, 26100, 26200])
        print(f"✅ Max Pain Test Passed: Calculated {mp}")

    def test_iv_calculation(self):
        """Test ATM IV Calculation"""
        spot = 26105 # Close to 26100
        iv = self.analyzer.calculate_atm_iv(self.mock_data, spot)
        # Expected: Avg of 11.5 and 13.5 = 12.5
        self.assertEqual(iv, 12.5)
        print(f"✅ IV Test Passed: Calculated {iv}")

    def test_oi_sentiment(self):
        """Test OI Change Sentiment"""
        spot = 26100
        # Call Chg: 5k + 10k - 1k = 14k
        # Put Chg: -2k + 15k + 20k = 33k
        # Put Chg (33k) > Call Chg (14k) * 1.5 -> BULLISH
        metrics = self.analyzer.analyze_oi_change(self.mock_data, spot)
        self.assertEqual(metrics["sentiment"], "BULLISH")
        print(f"✅ OI Sentiment Test Passed: {metrics['sentiment']}")


from datetime import datetime
from analysis_module.technical import Signal, SignalType

class TestSignalIntegration(unittest.TestCase):
    def setUp(self):
        self.agent = NiftyTradingAgent()
        self.agent.option_fetcher = MagicMock()
        self.agent.option_analyzer = OptionChainAnalyzer() # Use real logic

    def test_conflict_resolution_bullish_pcr(self):
        """Test that Bullish PCR resolves conflict in favor of LONG"""
        # Mock PCR > 1.2 (Bullish) via mock Option Data
        mock_oc_data = {
            "records": {"data": [
                 {"strikePrice": 26000, "CE": {"openInterest": 100}, "PE": {"openInterest": 200}}, # PCR=2.0
                 {"strikePrice": 26100, "CE": {"openInterest": 100}, "PE": {"openInterest": 200}}
            ]}
        }
        self.agent.option_fetcher.fetch_option_chain.return_value = mock_oc_data
        
        # Prepare conflicting objects
        # 1. Bullish Signal (e.g., Retest)
        bullish_sig = Signal(
            signal_type=SignalType.SUPPORT_BOUNCE,
            instrument="NIFTY",
            timeframe="5m",
            price_level=26000.0,
            entry_price=26010.0,
            stop_loss=25980.0,
            take_profit=26100.0,
            confidence=60.0,
            timestamp=datetime.now(),
            description="Bullish Bounce",
            risk_reward_ratio=2.0,
            volume_confirmed=True,
            momentum_confirmed=True
        )
        
        # 2. Bearish Signal (e.g., Engulfing) - Higher confidence to test override
        bearish_sig = Signal(
            signal_type=SignalType.BEARISH_ENGULFING,
            instrument="NIFTY",
            timeframe="5m",
            price_level=26050.0,
            entry_price=26040.0,
            stop_loss=26060.0,
            take_profit=25950.0,
            confidence=80.0, # Higher confidence
            timestamp=datetime.now(),
            description="Bearish Engulfing",
            risk_reward_ratio=2.0,
            volume_confirmed=True,
            momentum_confirmed=True
        )
        
        # Inject into analysis dict
        analysis_input = {
            "retest_signal": bullish_sig,
            "engulfing_signal": bearish_sig
        }
        
        nse_data_input = {"lastPrice": 26020, "price": 26020}
        
        # We need to ensure volume_check doesn't block it.
        # But looking at main.py, _generate_signals calls self.technical_analyzer only for breakout_signal logic?
        # Re-reading main.py:
        # lines 356 (breakout) -> checks volume_confirmed
        # lines 383 (retest) -> no extra checks inside _generate_signals logic itself usually?
        # Let's verify specific blocks in main.py logic for retest/engulfing.
        # It seems simple appending if confidence >= MIN.
        
        # Run
        signals = self.agent._generate_signals("NIFTY", analysis_input, nse_data_input)
        
        # Verify
        # Should have 1 signal
        self.assertEqual(len(signals), 1)
        # Should be the Bullish one despite lower confidence, because PCR=2.0 (Bullish)
        self.assertIn("SUPPORT_BOUNCE", signals[0]["signal_type"])
        print(f"\n✅ Conflict Resolution Passed: Kept {signals[0]['signal_type']} due to PCR")

if __name__ == '__main__':
    unittest.main()
