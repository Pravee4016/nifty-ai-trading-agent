"""
Adaptive Threshold Module
Adjusts RSI and ATR thresholds based on market volatility regime.
"""

import logging
from typing import Dict, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class AdaptiveThresholds:
    """
    Dynamically adjust technical thresholds based on market conditions.
    """
    
    @staticmethod
    def get_rsi_thresholds(vix: float = None, atr_percentile: float = None) -> Tuple[int, int]:
        """
        Get adaptive RSI thresholds based on volatility.
        
        Args:
            vix: India VIX value (if available)
            atr_percentile: ATR percentile (0-100)
        
        Returns:
            (rsi_long_threshold, rsi_short_threshold)
        """
        # Default values (normal market)
        rsi_long = 60
        rsi_short = 40
        
        # VIX-based adjustment (preferred)
        if vix is not None:
            if vix < 12:
                # Low volatility = sideways = tighter bands
                rsi_long = 55
                rsi_short = 45
                logger.debug(f"📊 Adaptive RSI: VIX {vix:.1f} (Low Volatility) → Tighter ({rsi_short}/{rsi_long})")
            elif vix > 18:
                # High volatility = trending = wider bands
                rsi_long = 65
                rsi_short = 35
                logger.debug(f"📊 Adaptive RSI: VIX {vix:.1f} (High Volatility) → Wider ({rsi_short}/{rsi_long})")
            else:
                # Normal
                logger.debug(f"📊 Adaptive RSI: VIX {vix:.1f} (Normal) → Default ({rsi_short}/{rsi_long})")
        
        # ATR percentile fallback (if VIX unavailable)
        elif atr_percentile is not None:
            if atr_percentile < 30:
                # Low volatility
                rsi_long = 55
                rsi_short = 45
                logger.debug(f"📊 Adaptive RSI: ATR %ile {atr_percentile:.1f} (Low Vol) → Tighter")
            elif atr_percentile > 70:
                # High volatility
                rsi_long = 65
                rsi_short = 35
                logger.debug(f"📊 Adaptive RSI: ATR %ile {atr_percentile:.1f} (High Vol) → Wider")
        
        return rsi_long, rsi_short
    
    @staticmethod
    def get_atr_threshold(df: pd.DataFrame, atr_period: int = 14) -> float:
        """
        Get adaptive ATR threshold as percentile of recent ATR values.
        
        Returns: ATR value at 60th percentile (tradeable volatility)
        """
        if df is None or len(df) < atr_period + 60:
            return 0.0
        
        # Calculate ATR history
        if "atr" not in df.columns:
            logger.warning("⚠️ ATR column not found in dataframe")
            return 0.0
            
        atr_history = df["atr"].iloc[-(atr_period + 60):-1]
        
        # Use 60th percentile as threshold
        atr_threshold = atr_history.quantile(0.60)
        
        logger.debug(f"📊 Adaptive ATR Threshold: {atr_threshold:.2f} (60th %ile)")
        return atr_threshold
    
    @staticmethod
    def calculate_atr_percentile(df: pd.DataFrame, current_atr: float, lookback: int = 60) -> float:
        """Calculate what percentile current ATR is in recent history."""
        if df is None or len(df) < lookback:
            return 50.0  # Default to median
        
        if "atr" not in df.columns:
            logger.warning("⚠️ ATR column not found in dataframe")
            return 50.0
            
        atr_history = df["atr"].iloc[-lookback:]
        percentile = (atr_history < current_atr).sum() / len(atr_history) * 100
        
        return percentile
    
    @staticmethod
    def is_market_volatile(vix: float = None, atr_percentile: float = None) -> bool:
        """
        Determine if market is in high volatility regime.
        
        Returns:
            True if volatile (VIX > 18 or ATR percentile > 70)
        """
        if vix is not None and vix > 18:
            return True
        if atr_percentile is not None and atr_percentile > 70:
            return True
        return False
    
    @staticmethod
    def is_market_choppy(vix: float = None, atr_percentile: float = None) -> bool:
        """
        Determine if market is in low volatility / choppy regime.
        
        Returns:
            True if choppy (VIX < 12 or ATR percentile < 30)
        """
        if vix is not None and vix < 12:
            return True
        if atr_percentile is not None and atr_percentile < 30:
            return True
        return False
