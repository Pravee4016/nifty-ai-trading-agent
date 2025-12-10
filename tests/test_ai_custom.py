
import unittest
from unittest.mock import MagicMock, patch
from ai_module.groq_analyzer import GroqAnalyzer

class TestGroqAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = GroqAnalyzer()

    @patch('ai_module.groq_analyzer.requests.Session.post')
    def test_analyze_signal_valid(self, mock_post):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"verdict": "STRONG_BUY", "confidence": 85, "reasoning": "Good setup", "risks": ["Vol"]}'
                }
            }]
        }
        mock_post.return_value = mock_response

        sig_data = {
            "signal_type": "BREAKOUT",
            "price_level": 26000,
            "entry_price": 26050,
            "confidence": 80,
            "risk_reward_ratio": 2.5
        }
        context = {
            "trend_direction": "UP",
            "price_above_vwap": True,
            "price_above_ema20": True
        }
        
        result = self.analyzer.analyze_signal(sig_data, context, {})
        
        self.assertIsNotNone(result)
        self.assertEqual(result['verdict'], "STRONG_BUY")
        self.assertEqual(result['confidence'], 85)

    def test_prompt_construction(self):
        sig_data = {"signal_type": "TEST", "entry_price": 100}
        context = {"trend_direction": "UP"}
        prompt = self.analyzer._construct_prompt(sig_data, context, {})
        self.assertIn("INSTRUMENT: NIFTY 50", prompt)
        self.assertIn("SIGNAL: TEST", prompt)

if __name__ == '__main__':
    unittest.main()
