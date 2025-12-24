"""
AI Factory Module
Provides abstraction layer for AI analyzer selection (Groq vs Vertex AI vs Hybrid).
"""

import logging
import random
from typing import Dict, Optional

from config.settings import (
    AI_PROVIDER,
    HYBRID_GROQ_WEIGHT,
    HYBRID_VERTEX_WEIGHT,
)

logger = logging.getLogger(__name__)


class HybridAnalyzer:
    """
    Hybrid AI analyzer that routes requests between Groq and Vertex AI.
    Useful for A/B testing to compare quality and performance.
    """
    
    def __init__(self):
        from ai_module.groq_analyzer import get_analyzer as get_groq
        from ai_module.vertex_analyzer import get_analyzer as get_vertex
        
        self.groq = get_groq()
        self.vertex = get_vertex()
        self.groq_weight = HYBRID_GROQ_WEIGHT
        self.vertex_weight = HYBRID_VERTEX_WEIGHT
        
        # Normalize weights
        total = self.groq_weight + self.vertex_weight
        self.groq_weight /= total
        self.vertex_weight /= total
        
        logger.info(
            f"🔀 Hybrid AI Mode | Groq: {self.groq_weight*100:.0f}% | "
            f"Vertex: {self.vertex_weight*100:.0f}%"
        )
    
    def analyze_signal(
        self,
        signal_data: Dict,
        market_context: Dict,
        technical_data: Dict
    ) -> Optional[Dict]:
        """Route to Groq or Vertex based on configured weights."""
        
        # Random selection based on weights
        if random.random() < self.groq_weight:
            logger.debug("🎲 Hybrid: Using Groq")
            result = self.groq.analyze_signal(signal_data, market_context, technical_data)
            if result:
                result["ai_provider"] = "GROQ"
            return result
        else:
            logger.debug("🎲 Hybrid: Attempting Vertex AI")
            try:
                result = self.vertex.analyze_signal(signal_data, market_context, technical_data)
                if result and "error" not in result:
                    result["ai_provider"] = "VERTEX"
                    return result
                else:
                    logger.warning(f"⚠️ Vertex AI returned error or empty result, falling back to Groq")
            except Exception as e:
                logger.error(f"❌ Vertex AI failed in Hybrid mode: {e}. Falling back to Groq.")
            
            # Fallback to Groq
            result = self.groq.analyze_signal(signal_data, market_context, technical_data)
            if result:
                result["ai_provider"] = "GROQ (Fallback)"
            return result
    
    def test_connection(self) -> bool:
        """Test both providers."""
        groq_ok = self.groq.test_connection()
        vertex_ok = self.vertex.test_connection()
        
        if not groq_ok and not vertex_ok:
            return False
        
        if not groq_ok:
            logger.warning("⚠️ Groq unavailable in hybrid mode - using Vertex only")
        if not vertex_ok:
            logger.warning("⚠️ Vertex unavailable in hybrid mode - using Groq only")
        
        return True
    
    def get_usage_stats(self) -> Dict:
        """Return combined stats from both providers."""
        return {
            "mode": "HYBRID",
            "groq_weight": f"{self.groq_weight*100:.0f}%",
            "vertex_weight": f"{self.vertex_weight*100:.0f}%",
            "groq_stats": self.groq.get_usage_stats(),
            "vertex_stats": self.vertex.get_usage_stats(),
        }


def get_analyzer(provider: str = None):
    """
    Factory function to get appropriate AI analyzer.
    
    Args:
        provider: AI provider to use (GROQ, VERTEX, HYBRID). 
                  If None, uses AI_PROVIDER from settings.
    
    Returns:
        AI analyzer instance with analyze_signal() method.
    
    Example:
        analyzer = get_analyzer()
        result = analyzer.analyze_signal(signal_data, market_context, technical_data)
    """
    provider = (provider or AI_PROVIDER).upper()
    
    if provider == "GROQ":
        logger.debug("🧠 Using Groq AI")
        from ai_module.groq_analyzer import get_analyzer as get_groq
        return get_groq()
    
    elif provider == "VERTEX":
        logger.debug("🧠 Using Vertex AI (Gemini)")
        from ai_module.vertex_analyzer import get_analyzer as get_vertex
        return get_vertex()
    
    elif provider == "HYBRID":
        logger.debug("🧠 Using Hybrid AI Mode")
        return HybridAnalyzer()
    
    else:
        logger.error(f"❌ Unknown AI provider: {provider}. Falling back to Groq.")
        from ai_module.groq_analyzer import get_analyzer as get_groq
        return get_groq()


# Singleton instance
_analyzer_instance = None


def get_default_analyzer():
    """
    Get singleton analyzer instance using configured provider.
    Cached for performance.
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = get_analyzer()
    return _analyzer_instance
