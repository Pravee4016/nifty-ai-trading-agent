"""
Confluence Detection Module
Identifies when multiple technical levels converge at the same price point.
Expert's core methodology for high-probability setups.
"""

import logging
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TechnicalLevels:
    """Support/Resistance levels and related metrics"""
    support_levels: List[float]
    resistance_levels: List[float]
    pivot: float
    pdh: float
    pdl: float
    atr: float
    volatility_score: float
    rsi_divergence: str = "NONE"
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    ema_50: float = 0.0
    r1_fib: float = 0.0
    s1_fib: float = 0.0
    r2_fib: float = 0.0
    s2_fib: float = 0.0
    confluence_zones: List[Dict] = None


def detect_confluence(
    price: float, 
    levels: TechnicalLevels,
    higher_tf_context: Dict,
    tolerance_pct: float = 0.002  # DEPRECATED: Now using absolute points
) -> Dict:
    """
    Detect confluence of multiple technical levels at current price.
    
    Expert's methodology: Identify when 2+ key levels converge within ±3 points.
    
    Args:
        price: Current price to check
        levels: TechnicalLevels object with all key levels
        higher_tf_context: Context with EMAs, VWAP, etc.
        tolerance_pct: DEPRECATED - Now using absolute points from settings
        
    Returns:
        Dict with:
            - confluence_count: Number of levels near price
            - level_names: List of confluent level names
            - confluence_score: 0-100 score based on quality
            - is_high_probability: True if 2+ levels converge
    """
    # CRITICAL FIX: Use absolute points, not percentage
    # Old: tolerance = price * 0.002 = 25000 * 0.002 = 50 points (TOO WIDE!)
    # New: tolerance = 3.0 points (precise confluence detection)
    from config.settings import CONFLUENCE_TOLERANCE_POINTS
    tolerance = CONFLUENCE_TOLERANCE_POINTS  # Absolute 3 points, not percentage

    confluent_levels = []
    
    # Check all key levels
    level_checks = {
        'PDH': levels.pdh,
        'PDL': levels.pdl,
        'Fib_R1': levels.r1_fib,
        'Fib_S1': levels.s1_fib,
        'Fib_R2': levels.r2_fib,
        'Fib_S2': levels.s2_fib,
        'Pivot': levels.pivot,
        'EMA20_5m': higher_tf_context.get('ema_20_5m', 0),
        'EMA50_5m': higher_tf_context.get('ema_50_5m', 0),
        'EMA50_15m': higher_tf_context.get('ema_50_15m', 0),
        'VWAP': higher_tf_context.get('vwap_5m', 0),
        'BB_Upper': higher_tf_context.get('bb_upper_5m', 0),
        'BB_Middle': higher_tf_context.get('bb_middle_5m', 0),
        'BB_Lower': higher_tf_context.get('bb_lower_5m', 0),
    }
    
    for name, level in level_checks.items():
        if level > 0 and abs(price - level) <= tolerance:
            confluent_levels.append({
                'name': name,
                'level': level,
                'distance': abs(price - level),
                'distance_pct': abs(price - level) / price * 100
            })
    
    # Calculate confluence score
    confluence_count = len(confluent_levels)
    
    # Base score
    if confluence_count == 0:
        confluence_score = 0
    elif confluence_count == 1:
        confluence_score = 30
    elif confluence_count == 2:
        confluence_score = 70  # Expert's threshold
    elif confluence_count == 3:
        confluence_score = 90
    else:
        confluence_score = 100
    
    # Bonus for key combinations (Expert's favorites)
    level_names = [l['name'] for l in confluent_levels]
    
    # PDH + Fib R1 (resistance confluence)
    if 'PDH' in level_names and 'Fib_R1' in level_names:
        confluence_score = min(100, confluence_score + 10)
        
    # PDL + Fib S1 (support confluence)
    if 'PDL' in level_names and 'Fib_S1' in level_names:
        confluence_score = min(100, confluence_score + 10)
        
    # EMA20 + Fib Level (dynamic support/resistance)
    if 'EMA20_5m' in level_names and any('Fib' in n for n in level_names):
        confluence_score = min(100, confluence_score + 10)
        
    # BB extreme + Fib level (oversold/overbought confluence)
    if ('BB_Upper' in level_names or 'BB_Lower' in level_names) and any('Fib' in n for n in level_names):
        confluence_score = min(100, confluence_score + 10)
    
    is_high_probability = confluence_count >= 2
    
    if confluence_count >= 2:
        logger.info(
            f"🎯 CONFLUENCE DETECTED | {confluence_count} levels | "
            f"Price: {price:.2f} | Levels: {', '.join(level_names)}"
        )
    
    return {
        'confluence_count': confluence_count,
        'level_names': level_names,
        'confluent_levels': confluent_levels,
        'confluence_score': confluence_score,
        'is_high_probability': is_high_probability,
        'tolerance_used': tolerance
    }
