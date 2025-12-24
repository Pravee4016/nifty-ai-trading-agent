"""
Vertex AI Gemini Analyzer Module
Uses Gemini 1.5 Pro via Vertex AI for trading signal analysis.
Drop-in replacement for Groq with enhanced capabilities.
"""

import json
import logging
from typing import Dict, Optional
from datetime import datetime

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    logging.warning("⚠️ Vertex AI SDK not installed. Run: pip install google-cloud-aiplatform")

from config.settings import (
    VERTEX_PROJECT_ID,
    VERTEX_LOCATION,
    VERTEX_MODEL,
)

logger = logging.getLogger(__name__)


class VertexAnalyzer:
    """Gemini-based signal analyzer using Vertex AI."""
    
    def __init__(self):
        self.project_id = VERTEX_PROJECT_ID
        self.location = VERTEX_LOCATION
        self.model_name = VERTEX_MODEL
        
        if not VERTEX_AVAILABLE:
            logger.error("❌ Vertex AI SDK not available")
            self.enabled = False
            return
        
        if not self.project_id or "YOUR_PROJECT" in self.project_id:
            logger.warning("⚠️ Vertex AI Project ID not configured. AI disabled.")
            self.enabled = False
            return
        
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            
            # Initialize Gemini model
            self.model = GenerativeModel(
                self.model_name,
                generation_config=GenerationConfig(
                    temperature=0.3,  # Lower than Groq for more consistent analysis
                    max_output_tokens=500,
                    response_mime_type="application/json",  # Force JSON output
                )
            )
            
            self.enabled = True
            logger.info(f"🧠 Vertex AI Gemini Initialized | Model: {self.model_name} | Project: {self.project_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vertex AI: {str(e)}")
            self.enabled = False
    
    def analyze_signal(
        self,
        signal_data: Dict,
        market_context: Dict,
        technical_data: Dict
    ) -> Optional[Dict]:
        """
        Analyze a trading signal using Gemini 1.5 Pro.
        
        Returns:
            Dict containing:
            - reasoning (str): Natural language explanation
            - confidence (int): 0-100 score
            - risks (List[str]): Potential risks
            - verdict (str): "STRONG_BUY", "CAUTIOUS_BUY", "STRONG_SELL", "CAUTIOUS_SELL", "PASS"
        """
        if not self.enabled:
            return {"error": "Vertex AI Disabled"}
        
        try:
            # Construct prompt using same logic as Groq
            system_prompt = self._get_system_prompt()
            user_prompt = self._construct_prompt(signal_data, market_context, technical_data)
            
            # Combine system and user prompts (Gemini doesn't have separate system role)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            start_time = datetime.now()
            
            # Generate content
            response = self.model.generate_content(full_prompt)
            
            latency = (datetime.now() - start_time).total_seconds()
            
            # Parse JSON response
            result_text = response.text.strip()
            
            # Try to extract JSON if wrapped in markdown
            if result_text.startswith("```json"):
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif result_text.startswith("```"):
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            parsed_result = json.loads(result_text)
            
            logger.info(
                f"🤖 Vertex AI Analysis Complete ({latency:.2f}s) | "
                f"Verdict: {parsed_result.get('verdict')} | "
                f"Confidence: {parsed_result.get('confidence')}%"
            )
            
            return parsed_result
        
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse Vertex AI JSON response: {str(e)}")
            logger.error(f"   Raw response: {response.text[:200]}...")
            return None
        
        except Exception as e:
            logger.error(f"❌ Vertex AI Analysis Failed: {str(e)}")
            return None
    
    def test_connection(self) -> bool:
        """Test if Vertex AI is accessible and configured correctly."""
        if not self.enabled:
            return False
        
        try:
            # Simple test prompt
            test_prompt = "Respond with valid JSON: {\"status\": \"ok\"}"
            response = self.model.generate_content(test_prompt)
            
            # Check if we got a response
            if response and response.text:
                logger.info("✅ Vertex AI connection test successful")
                return True
            
            return False
        
        except Exception as e:
            logger.warning(f"⚠️ Vertex AI Connection Test Failed: {e}")
            return False
    
    def get_usage_stats(self) -> Dict:
        """Return usage statistics for the AI module."""
        return {
            "enabled": self.enabled,
            "provider": "Vertex AI",
            "model": self.model_name,
            "project": self.project_id,
            "location": self.location,
            "requests_today": 0,  # Not tracked locally - available in GCP console
        }
    
    def _get_system_prompt(self) -> str:
        """Same system prompt as Groq for consistency."""
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
- Medium Confidence (60-80): Good structure but mixed trend.
- Fail (<50): Counter-trend without reversal structure, or poor metrics.
- DIRECTION: Ensure verdict matches signal direction (SELL for SHORT, BUY for LONG)."""
    
    def _construct_prompt(self, sig: Dict, context: Dict, tech: Dict) -> str:
        """Construct the dynamic prompt with signal details."""
        
        # Extract signal info
        signal_type = sig.get('signal_type', 'UNKNOWN')
        price = sig.get('price_level', 0)
        trend_15m = context.get('trend_direction', 'FLAT')
        entry_price = float(sig.get('entry_price', 0))
        stop_loss = float(sig.get('stop_loss', 0))
        
        # Determine signal direction from entry vs stop loss
        if stop_loss > entry_price:
            direction = "SHORT"
        elif stop_loss < entry_price:
            direction = "LONG"
        else:
            # Fallback: infer from signal type
            if any(x in signal_type.upper() for x in ["BEARISH", "RESISTANCE", "SHORT", "BREAKDOWN"]):
                direction = "SHORT"
            else:
                direction = "LONG"
        
        mtf_data = (
            f"15m Trend: {trend_15m}\n"
            f"Rel to VWAP: {'Above' if context.get('price_above_vwap') else 'Below'}\n"
            f"Rel to EMA20: {'Above' if context.get('price_above_ema20') else 'Below'}"
        )
        
        return f"""
ANALYZE THIS TRADE SETUP:

INSTRUMENT: NIFTY 50
SIGNAL: {signal_type}
DIRECTION: {direction}  ← CRITICAL: Use SELL verdicts for SHORT, BUY verdicts for LONG
LEVEL: {price}
ENTRY: {entry_price}
STOP LOSS: {stop_loss}

TECHNICAL CONTEXT:
{mtf_data}

SIGNAL METRICS:
- Confidence: {sig.get('confidence')}%
- R:R Ratio: {sig.get('risk_reward_ratio', 0):.2f}
- Description: {sig.get('description')}

NOTE: This is an INDEX instrument - volume data is not available/relevant.

CRITICAL: Your verdict MUST match the DIRECTION above:
- If DIRECTION is SHORT → use STRONG_SELL or CAUTIOUS_SELL (NOT BUY)
- If DIRECTION is LONG → use STRONG_BUY or CAUTIOUS_BUY (NOT SELL)

Evaluate based on Multi-Timeframe alignment and Market Structure.
"""


# Singleton access
_analyzer_instance = None


def get_analyzer():
    """Return singleton Vertex AI analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = VertexAnalyzer()
    return _analyzer_instance
