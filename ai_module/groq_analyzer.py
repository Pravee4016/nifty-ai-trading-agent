"""
Groq AI Analyzer Module
Uses LLaMA 3 70B via Groq API to provide "Hedge Fund Analyst" reasoning for technical signals.
"""

import os
import json
import logging
import requests
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from config.settings import (
    GROQ_API_KEY, 
    GROQ_MODEL, 
    GROQ_TEMPERATURE, 
    GROQ_MAX_TOKENS
)

logger = logging.getLogger(__name__)

class GroqAnalyzer:
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = GROQ_MODEL
        
        if not self.api_key or "YOUR_KEY" in self.api_key:
            logger.warning("⚠️ Groq API Key not found. AI Analysis disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"🧠 Groq AI Initialized | Model: {self.model}")

        # Setup resilient session
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def analyze_signal(
        self, 
        signal_data: Dict, 
        market_context: Dict, 
        technical_data: Dict
    ) -> Dict:
        """
        Analyze a trading signal using LLaMA 3.
        
        Returns:
            Dict containing:
            - reasoning (str): Natural language explanation
            - confidence (int): 0-100 score
            - risks (List[str]): Potential risks
            - verdict (str): "STRONG BUY", "CAUTIOUS BUY", "PASS"
        """
        if not self.enabled:
            return {"error": "AI Disabled"}

        try:
            prompt = self._construct_prompt(signal_data, market_context, technical_data)
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                "temperature": GROQ_TEMPERATURE,
                "max_tokens": GROQ_MAX_TOKENS,
                "response_format": {"type": "json_object"}
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            start_time = datetime.now()
            response = self.session.post(
                self.api_url, 
                json=payload, 
                headers=headers, 
                timeout=10
            ) 
            response.raise_for_status()
            
            latency = (datetime.now() - start_time).total_seconds()
            result = response.json()
            
            ai_content = result['choices'][0]['message']['content']
            parsed_result = json.loads(ai_content)
            
            logger.info(f"🤖 AI Analysis Complete ({latency:.2f}s) | Confidence: {parsed_result.get('confidence')}%")
            
            return parsed_result

        except Exception as e:
            logger.error(f"❌ Groq Analysis Failed: {str(e)}")
            return None

    def test_connection(self) -> bool:
        """Test if Groq API is reachable and key is valid."""
        if not self.enabled:
            return False
            
        try:
            # Lightweight test call (list models)
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = self.session.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"⚠️ Groq Connection Test Failed: {e}")
            return False

    def get_usage_stats(self) -> Dict:
        """Return usage statistics for the AI module."""
        return {
            "enabled": self.enabled,
            "model": self.model,
            "requests_today": 0, # TODO: Implement real tracking if needed
            "tokens_used": 0
        }

    def _get_system_prompt(self) -> str:
        return """You are a Senior Hedge Fund Technical Analyst. 
Your job is to VALIDATE algorithmic trading signals for NIFTY 50.
You are skeptical, risk-averse, and focus on CONFLUENCE.

OUTPUT FORMAT (JSON):
{
    "verdict": "STRONG_BUY" | "CAUTIOUS_BUY" | "STRONG_SELL" | "CAUTIOUS_SELL" | "PASS",
    "confidence": <0-100 integer>,
    "reasoning": "<Concise 2-sentence explanation focusing on WHY this works or fails>",
    "risks": ["<Risk 1>", "<Risk 2>"]
}

SCORING RULES:
- High Confidence (>80): Needs Trend Alignment + Structure Breakout + Good R:R.
- Medium Confidence (60-80): Good structure but mixed trend or low volume.
- Fail (<50): Counter-trend without reversal structure, or poor metrics.
- DIRECTION: Ensure verdict matches signal direction (SELL for Bearish, BUY for Bullish)."""

    def forecast_market_outlook(self, daily_summary_text: str) -> Dict:
        """
        Generate a market outlook forecast based on the day's summary.
        
        Args:
            daily_summary_text (str): A text summary of the day's price action and signals.
            
        Returns:
            Dict: {
                "outlook": "BULLISH" | "BEARISH" | "NEUTRAL",
                "confidence": int,
                "summary": str
            }
        """
        if not self.enabled:
            return {"outlook": "NEUTRAL", "confidence": 0, "summary": "AI Disabled"}

        try:
            system_prompt = """You are a Market Strategist. 
Analyze the provided end-of-day market summary and forecast the outlook for TOMORROW.
Consider trend, support/resistance tests, and overall sentiment.
OUTPUT JSON: {"outlook": "BULLISH"|"BEARISH"|"NEUTRAL", "confidence": 0-100, "summary": "One sentence outlook."}"""

            user_prompt = f"MARKET SUMMARY:\n{daily_summary_text}\n\nFORECAST THE NEXT SESSION:"

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 150,
                "response_format": {"type": "json_object"}
            }

            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = self.session.post(
                self.api_url, 
                json=payload, 
                headers=headers, 
                timeout=10
            ) 
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            return json.loads(content)

        except Exception as e:
            logger.error(f"❌ Forecast failed: {e}")
            return {"outlook": "NEUTRAL", "confidence": 0, "summary": "Forecast Error"}


    def _construct_prompt(self, sig: Dict, context: Dict, tech: Dict) -> str:
        """Construct the dynamic prompt with signal details."""
        
        # safely access keys
        signal_type = sig.get('signal_type', 'UNKNOWN')
        price = sig.get('price_level', 0)
        trend_15m = context.get('trend_direction', 'FLAT')
        
        mtf_data = (
            f"15m Trend: {trend_15m}\n"
            f"Rel to VWAP: {'Above' if context.get('price_above_vwap') else 'Below'}\n"
            f"Rel to EMA20: {'Above' if context.get('price_above_ema20') else 'Below'}"
        )
        
        option_data = "N/A"
        # If we have option metrics passed in "tech" or "context"
        # Adapted based on usage in main.py
        
        return f"""
ANALYZE THIS TRADE SETUP:

INSTRUMENT: NIFTY 50
SIGNAL: {signal_type}
LEVEL: {price}
CURRENT PRICE: {sig.get('entry_price')}

TECHNICAL CONTEXT:
{mtf_data}

SIGNAL METRICS:
- Confidence: {sig.get('confidence')}%
- R:R Ratio: {sig.get('risk_reward_ratio', 0):.2f}
- Volume Surge: {sig.get('volume_confirmed')}
- Description: {sig.get('description')}

Evaluate based on Multi-Timeframe alignment and Market Structure.
"""

# Singleton access
_analyzer_instance = None

def get_analyzer():
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = GroqAnalyzer()
    return _analyzer_instance
