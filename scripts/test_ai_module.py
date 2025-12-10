
import os
import sys
import logging
import json
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_module.groq_analyzer import GroqAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestAI")

def test_ai_module_mock():
    logger.info("--- Testing AI Module (Mock) ---")
    
    with patch('ai_module.groq_analyzer.requests.Session.post') as mock_post:
        # Mock Successful Response
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Valid JSON response expected from LLM
        llm_output = {
            "verdict": "STRONG_BUY",
            "confidence": 88,
            "reasoning": "Price broke 25800 resistance with High Volume.",
            "risks": ["RSI Overbought", "News Event"]
        }
        
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(llm_output)
                    }
                }
            ]
        }
        
        mock_post.return_value = mock_response
        
        analyzer = GroqAnalyzer()
        
        # Mock Signal Data
        signal = {
            "signal_type": "BULLISH_BREAKOUT",
            "price_level": 25800,
            "entry_price": 25810,
            "confidence": 80,
            "risk_reward_ratio": 2.5,
            "volume_confirmed": True,
            "description": "Breakout above PDH"
        }
        
        context = {
            "trend_direction": "UP",
            "price_above_vwap": True,
            "price_above_ema20": True
        }
        
        logger.info("Sending Signal for Analysis...")
        result = analyzer.analyze_signal(signal, context, {})
        
        if result:
            logger.info("✅ Analysis Successful")
            logger.info(f"   Verdict: {result['verdict']}")
            logger.info(f"   Confidence: {result['confidence']}%")
            logger.info(f"   Reasoning: {result['reasoning']}")
            
            if result['verdict'] == "STRONG_BUY" and result['confidence'] == 88:
                 logger.info("✅ Parsing Logic Verified")
            else:
                 logger.error("❌ Parsing mismatch")
        else:
            logger.error("❌ Analysis Failed (returned None)")

if __name__ == "__main__":
    test_ai_module_mock()
