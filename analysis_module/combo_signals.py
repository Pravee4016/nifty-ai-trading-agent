"""
MACD + RSI + Bollinger Bands Combo Signal Evaluator
Analyzes confluence of multiple indicators for signal strength.
"""

import logging
from typing import Dict
import pandas as pd

logger = logging.getLogger(__name__)


class MACDRSIBBCombo:
    """
    Evaluates signal strength based on MACD, RSI, and Bollinger Bands confluence.
    
    Strength Levels:
        - STRONG: All 3 conditions met + extreme RSI
        - MEDIUM: 2 out of 3 conditions met
        - WEAK: Only 1 condition met
        - INVALID: No conditions met or opposing signals
    """
    
    def __init__(self):
        logger.info("📊 MACDRSIBBCombo initialized")
    
    def evaluate_signal(
        self, 
        df: pd.DataFrame, 
        direction_bias: str,  # From EMA crossover or pattern direction
        technical_context: Dict
    ) -> Dict:
        """
        Evaluate signal strength based on indicator confluence.
        
        Args:
            df: OHLCV DataFrame with indicators
            direction_bias: "BULLISH" or "BEARISH" from EMA crossover or pattern
            technical_context: Dict with MACD, RSI, BB data
        
        Returns:
            {
                "strength": str,  # "STRONG", "MEDIUM", "WEAK", "INVALID"
                "score": int,     # 0-3 (number of conditions met)
                "conditions": {
                    "bb_favorable": bool,
                    "rsi_favorable": bool,
                    "macd_favorable": bool
                },
                "bb_position": float,  # 0.0 to 1.0
                "details": str  # Human-readable explanation
            }
        """
        try:
            current_price = df['close'].iloc[-1]
            
            # Extract indicators from context
            macd_data = technical_context.get("macd", {})
            rsi = technical_context.get("rsi_5", 50)
            bb_upper = technical_context.get("bb_upper", 0)
            bb_lower = technical_context.get("bb_lower", 0)
            
            # Get previous RSI for trend detection
            rsi_prev = 50
            if len(df) >= 2:
                # Calculate RSI series if needed
                if 'rsi' not in df.columns:
                    from config.settings import RSI_PERIOD
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                
                if not df['rsi'].isna().iloc[-2]:
                    rsi_prev = df['rsi'].iloc[-2]
            
            # Calculate Bollinger Band position (0 = lower band, 1 = upper band)
            bb_position = 0.5
            if bb_upper > bb_lower > 0:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                bb_position = max(0.0, min(1.0, bb_position))  # Clamp to [0, 1]
            
            # Evaluate conditions
            conditions = {}
            conditions_met = 0
            
            if direction_bias == "BULLISH" or direction_bias == "LONG":
                # Condition 1: Price in lower 35% of Bollinger Bands (oversold zone)
                conditions["bb_favorable"] = bb_position < 0.35
                
                # Condition 2: RSI < 40 and RISING (momentum building)
                conditions["rsi_favorable"] = rsi < 40 and rsi > rsi_prev
                
                # Condition 3:MACD histogram > 0 OR bullish crossover
                conditions["macd_favorable"] = (
                    macd_data.get("histogram", 0) > 0 or 
                    macd_data.get("crossover") == "BULLISH"
                )
                
            elif direction_bias == "BEARISH" or direction_bias == "SHORT":
                # Condition 1: Price in upper 35% of Bollinger Bands (overbought zone)
                conditions["bb_favorable"] = bb_position > 0.65
                
                # Condition 2: RSI > 60 and FALLING (momentum weakening)
                conditions["rsi_favorable"] = rsi > 60 and rsi < rsi_prev
                
                # Condition 3: MACD histogram < 0 OR bearish crossover
                conditions["macd_favorable"] = (
                    macd_data.get("histogram", 0) < 0 or 
                    macd_data.get("crossover") == "BEARISH"
                )
            
            else:  # NEUTRAL
                return {
                    "strength": "INVALID",
                    "score": 0,
                    "conditions": {},
                    "bb_position": bb_position,
                    "details": "No directional bias"
                }
            
            # Count conditions met
            conditions_met = sum(conditions.values())
            
            # Determine strength
            strength = "INVALID"
            details = ""
            
            # Normalize direction for checks
            is_bullish = direction_bias in ["BULLISH", "LONG"]
            
            if is_bullish:
                if conditions_met >= 3 and rsi < 30:
                    strength = "STRONG"
                    details = f"All conditions met + RSI extreme ({rsi:.1f})"
                elif conditions_met >= 2:
                    strength = "MEDIUM"
                    details = f"{conditions_met}/3 conditions met"
                elif conditions_met == 1:
                    strength = "WEAK"
                    details = f"Only {conditions_met}/3 condition met"
                else:
                    strength = "INVALID"
                    details = "No conditions met"
            
            else:  # BEARISH/SHORT
                if conditions_met >= 3 and rsi > 70:
                    strength = "STRONG"
                    details = f"All conditions met + RSI extreme ({rsi:.1f})"
                elif conditions_met >= 2:
                    strength = "MEDIUM"
                    details = f"{conditions_met}/3 conditions met"
                elif conditions_met == 1:
                    strength = "WEAK"
                    details = f"Only {conditions_met}/3 condition met"
                else:
                    strength = "INVALID"
                    details = "No conditions met"
            
            # Log the evaluation
            logger.info(
                f"📊 Combo Signal: {direction_bias} | {strength} | "
                f"Score: {conditions_met}/3 | BB: {bb_position:.2f} | RSI: {rsi:.1f} | "
                f"MACD Hist: {macd_data.get('histogram', 0):.2f}"
            )
            
            return {
                "strength": strength,
                "score": conditions_met,
                "conditions": conditions,
                "bb_position": round(bb_position, 2),
                "details": details
            }
        
        except Exception as e:
            logger.error(f"Error evaluating combo signal: {e}")
            return {
                "strength": "INVALID",
                "score": 0,
                "conditions": {},
                "bb_position": 0.5,
                "details": f"Error: {str(e)}"
            }
