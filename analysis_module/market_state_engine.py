"""
Market State Engine
Determines if market conditions are suitable for scalping

States:
- CHOPPY: No trading (capital protection)
- TRANSITION: Selective trading (high-momentum only)
- EXPANSIVE: Normal trading (all strategies active)
"""

import logging
import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """Market state for scalping suitability."""
    CHOPPY = "CHOPPY"
    TRANSITION = "TRANSITION"
    EXPANSIVE = "EXPANSIVE"


class MarketStateEngine:
    """
    Evaluates market state for scalping suitability.
    
    Uses multiple metrics to determine if market is:
    - CHOPPY: Overlapping, non-directional (no trades)
    - TRANSITION: Breaking from compression (selective trades)
    - EXPANSIVE: Strong directional movement (full trading)
    """
    
    def __init__(
        self,
        choppy_range_threshold: float = 20.0,
        choppy_wick_ratio: float = 0.55,
        choppy_vwap_crosses: int = 4,
        expansion_threshold: float = 12.0,
        expansive_range_threshold: float = 25.0
    ):
        """
        Initialize Market State Engine.
        
        Args:
            choppy_range_threshold: Max 6-candle range for CHOPPY (points)
            choppy_wick_ratio: Min avg wick ratio for CHOPPY
            choppy_vwap_crosses: Min VWAP crosses for CHOPPY
            expansion_threshold: Min follow-through for expansion (points)
            expansive_range_threshold: Min range for EXPANSIVE state (points)
        """
        self.choppy_range_threshold = choppy_range_threshold
        self.choppy_wick_ratio = choppy_wick_ratio
        self.choppy_vwap_crosses = choppy_vwap_crosses
        self.expansion_threshold = expansion_threshold
        self.expansive_range_threshold = expansive_range_threshold
        
        logger.info("✅ Market State Engine initialized")
    
    def evaluate_state(
        self,
        df: pd.DataFrame,
        vwap_series: pd.Series = None
    ) -> Dict:
        """
        Evaluate current market state.
        
        Args:
            df: OHLCV dataframe (last 20+ candles)
            vwap_series: Optional VWAP values
            
        Returns:
            {
                "state": MarketState,
                "confidence": float (0-1),
                "reasons": List[str],
                "metrics": Dict[str, any]
            }
        """
        if len(df) < 10:
            logger.warning("Insufficient data for state evaluation")
            return self._default_state()
        
        # Check CHOPPY conditions (any 2 of 4)
        choppy_score, choppy_metrics = self._check_choppy_conditions(df, vwap_series)
        
        if choppy_score >= 2:
            reasons = self._get_choppy_reasons(choppy_metrics)
            return {
                "state": MarketState.CHOPPY,
                "confidence": choppy_score / 4.0,
                "reasons": reasons,
                "metrics": choppy_metrics
            }
        
        # Check EXPANSIVE conditions
        expansive_score, expansive_metrics = self._check_expansive_conditions(df)
        
        if expansive_score >= 2:
            reasons = self._get_expansive_reasons(expansive_metrics)
            return {
                "state": MarketState.EXPANSIVE,
                "confidence": expansive_score / 3.0,
                "reasons": reasons,
                "metrics": expansive_metrics
            }
        
        # Default to TRANSITION (between CHOPPY and EXPANSIVE)
        return {
            "state": MarketState.TRANSITION,
            "confidence": 0.5,
            "reasons": ["Market breaking from compression"],
            "metrics": {
                "choppy_score": choppy_score,
                "expansive_score": expansive_score
            }
        }
    
    def _check_choppy_conditions(
        self,
        df: pd.DataFrame,
        vwap_series: pd.Series = None
    ) -> Tuple[int, Dict]:
        """
        Check for CHOPPY market conditions.
        
        Returns:
            (score, metrics) where score is 0-4 based on met conditions
        """
        score = 0
        metrics = {}
        
        # NEW Condition 0: Velocity-based choppy detection (slow grind)
        # Catches ranges that look big in points but took hours to form
        if len(df) >= 20:
            last_20 = df.tail(20)
            range_20 = last_20['high'].max() - last_20['low'].min()
            
            # Calculate time span in minutes (assuming 5m candles)
            time_span_minutes = len(last_20) * 5
            time_span_hours = time_span_minutes / 60.0
            
            # Calculate velocity (points per hour)
            velocity = range_20 / time_span_hours if time_span_hours > 0 else 0
            metrics["velocity_pts_per_hour_20"] = velocity
            
            # If velocity < 15 pts/hour over 20 candles, it's choppy (slow grind)
            # Example: 40pts in 100min (1.67hr) = 24 pts/hr -> NOT choppy
            #          40pts in 180min (3hr) = 13.3 pts/hr -> CHOPPY!
            if velocity < 15.0 and range_20 > 25:
                score += 1
                metrics["slow_grind_choppy"] = True
            else:
                metrics["slow_grind_choppy"] = False
        else:
            metrics["velocity_pts_per_hour_20"] = 0
            metrics["slow_grind_choppy"] = False
        
        # Condition 1: Range Compression
        # Last 6 candles have total range < 20 points
        last_6 = df.tail(6)
        total_range = last_6['high'].max() - last_6['low'].min()
        metrics["range_6_candles"] = total_range
        
        if total_range < self.choppy_range_threshold:
            score += 1
            metrics["range_compressed"] = True
        else:
            metrics["range_compressed"] = False
        
        # Condition 2: Wick Dominance
        # Average wick-to-candle ratio > 55%
        avg_wick_ratio = self._calculate_wick_ratio(last_6)
        metrics["avg_wick_ratio"] = avg_wick_ratio
        
        if avg_wick_ratio > self.choppy_wick_ratio:
            score += 1
            metrics["wick_dominant"] = True
        else:
            metrics["wick_dominant"] = False
        
        # Condition 3: VWAP Magnet
        # Price crosses VWAP 4+ times in last 10 candles
        if vwap_series is not None and len(vwap_series) >= 10:
            vwap_crosses = self._count_vwap_crosses(df.tail(10), vwap_series.tail(10))
            metrics["vwap_crosses"] = vwap_crosses
            
            if vwap_crosses >= self.choppy_vwap_crosses:
                score += 1
                metrics["vwap_magnet"] = True
            else:
                metrics["vwap_magnet"] = False
        else:
            metrics["vwap_crosses"] = 0
            metrics["vwap_magnet"] = False
        
        # Condition 4: Expansion Failure
        # No sustained follow-through after breaks
        has_expansion = self._check_expansion_follow_through(df)
        metrics["has_expansion"] = has_expansion
        
        if not has_expansion:
            score += 1
            metrics["expansion_failed"] = True
        else:
            metrics["expansion_failed"] = False
        
        return score, metrics
    
    def _check_expansive_conditions(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Check for EXPANSIVE market conditions.
        
        Returns:
            (score, metrics) where score is 0-3 based on met conditions
        """
        score = 0
        metrics = {}
        
        last_10 = df.tail(10)
        
        # NEW: Velocity check - penalize slow ranges
        # Calculate time-adjusted velocity
        time_span_minutes = len(last_10) * 5  # 5m candles
        time_span_hours = time_span_minutes / 60.0
        total_range = last_10['high'].max() - last_10['low'].min()
        
        velocity = total_range / time_span_hours if time_span_hours > 0 else 0
        metrics["velocity_pts_per_hour"] = velocity
        
        # If velocity is too low, it's not truly expansive
        # Deduct 1 point if velocity < 20 pts/hour
        if velocity < 20.0 and total_range > 20:
            score -= 1  # Penalty for slow grinding
            metrics["slow_velocity_penalty"] = True
        
        # Condition 1: Large candle bodies (low wick ratio)
        avg_wick_ratio = self._calculate_wick_ratio(last_10)
        metrics["avg_wick_ratio"] = avg_wick_ratio
        
        if avg_wick_ratio < 0.35:  # Bodies > 65% of candle
            score += 1
            metrics["large_bodies"] = True
        else:
            metrics["large_bodies"] = False
        
        # Condition 2: Directional follow-through
        has_follow_through = self._check_directional_follow_through(last_10)
        metrics["has_follow_through"] = has_follow_through
        
        if has_follow_through:
            score += 1
        
        # Condition 3: Sustained range expansion
        total_range = last_10['high'].max() - last_10['low'].min()
        metrics["range_10_candles"] = total_range
        
        if total_range >= self.expansive_range_threshold:
            score += 1
            metrics["range_expanded"] = True
        else:
            metrics["range_expanded"] = False
        
        return score, metrics
    
    def _calculate_wick_ratio(self, df: pd.DataFrame) -> float:
        """
        Calculate average wick-to-candle ratio.
        
        Wick ratio = (upper_wick + lower_wick) / total_range
        """
        ratios = []
        
        for _, candle in df.iterrows():
            high = candle['high']
            low = candle['low']
            open_price = candle['open']
            close = candle['close']
            
            total_range = high - low
            if total_range == 0:
                continue
            
            body_high = max(open_price, close)
            body_low = min(open_price, close)
            
            upper_wick = high - body_high
            lower_wick = body_low - low
            
            wick_ratio = (upper_wick + lower_wick) / total_range
            ratios.append(wick_ratio)
        
        return np.mean(ratios) if ratios else 0.0
    
    def _count_vwap_crosses(
        self,
        df: pd.DataFrame,
        vwap_series: pd.Series
    ) -> int:
        """Count how many times price crosses VWAP."""
        crosses = 0
        
        for i in range(1, len(df)):
            prev_close = df.iloc[i-1]['close']
            curr_close = df.iloc[i]['close']
            prev_vwap = vwap_series.iloc[i-1]
            curr_vwap = vwap_series.iloc[i]
            
            # Check if price crossed VWAP
            prev_above = prev_close > prev_vwap
            curr_above = curr_close > curr_vwap
            
            if prev_above != curr_above:
                crosses += 1
        
        return crosses
    
    def _check_expansion_follow_through(self, df: pd.DataFrame) -> bool:
        """
        Check if recent breaks show follow-through expansion.
        
        Returns True if any recent break expanded 12-15 points within 2 candles.
        """
        if len(df) < 5:
            return False
        
        # Look for breaks in last 5 candles
        for i in range(len(df) - 4, len(df) - 1):
            candle = df.iloc[i]
            prev_candles = df.iloc[max(0, i-5):i]
            
            # Check if this candle broke recent high/low
            broke_high = candle['high'] > prev_candles['high'].max()
            broke_low = candle['low'] < prev_candles['low'].min()
            
            if broke_high or broke_low:
                # Check follow-through in next 1-2 candles
                next_candles = df.iloc[i+1:min(i+3, len(df))]
                
                if broke_high:
                    extension = next_candles['high'].max() - candle['high']
                else:
                    extension = candle['low'] - next_candles['low'].min()
                
                if extension >= self.expansion_threshold:
                    return True
        
        return False
    
    def _check_directional_follow_through(self, df: pd.DataFrame) -> bool:
        """
        Check if candles show sustained directional movement.
        
        Returns True if 3+ consecutive candles move in same direction.
        """
        if len(df) < 3:
            return False
        
        # Check bullish sequences
        bullish_count = 0
        bearish_count = 0
        
        for _, candle in df.iterrows():
            if candle['close'] > candle['open']:
                bullish_count += 1
                bearish_count = 0
            else:
                bearish_count += 1
                bullish_count = 0
            
            if bullish_count >= 3 or bearish_count >= 3:
                return True
        
        return False
    
    def _get_choppy_reasons(self, metrics: Dict) -> List[str]:
        """Generate human-readable reasons for CHOPPY state."""
        reasons = []
        
        if metrics.get("slow_grind_choppy"):
            reasons.append(f"Slow grind: {metrics['velocity_pts_per_hour_20']:.1f} pts/hr")
        
        if metrics.get("range_compressed"):
            reasons.append(f"Range compressed: {metrics['range_6_candles']:.1f}pts")
        
        if metrics.get("wick_dominant"):
            reasons.append(f"Wick dominance: {metrics['avg_wick_ratio']:.0%}")
        
        if metrics.get("vwap_magnet"):
            reasons.append(f"VWAP magnet: {metrics['vwap_crosses']} crosses")
        
        if metrics.get("expansion_failed"):
            reasons.append("No follow-through expansion")
        
        return reasons
    
    def _get_expansive_reasons(self, metrics: Dict) -> List[str]:
        """Generate human-readable reasons for EXPANSIVE state."""
        reasons = []
        
        if metrics.get("large_bodies"):
            reasons.append(f"Large bodies (wick: {metrics['avg_wick_ratio']:.0%})")
        
        if metrics.get("has_follow_through"):
            reasons.append("Directional follow-through confirmed")
        
        if metrics.get("range_expanded"):
            reasons.append(f"Range expansion: {metrics['range_10_candles']:.1f}pts")
        
        return reasons
    
    def _default_state(self) -> Dict:
        """Return default TRANSITION state for edge cases."""
        return {
            "state": MarketState.TRANSITION,
            "confidence": 0.3,
            "reasons": ["Insufficient data for full evaluation"],
            "metrics": {}
        }
