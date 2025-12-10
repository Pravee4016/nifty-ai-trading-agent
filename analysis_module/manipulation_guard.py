"""
Manipulation Guard (Circuit Breaker)
Protects against Flash Crashes, Freak Trades, and Expiry Manipulation.
"""

import logging
from datetime import datetime, time
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

from config.settings import (
    MAX_1MIN_MOVE_PCT,
    EXPIRY_STOP_TIME,
    VIX_PANIC_LEVEL,
    VIX_LOW_LEVEL,
    TIME_ZONE
)

logger = logging.getLogger(__name__)

class CircuitBreaker:
    def __init__(self):
        self.triggered = False
        self.trigger_reason = None
        self.trigger_time = None
        self.pause_duration = 0
        self.last_vix = 0.0

    def check_market_integrity(self, df_5m: pd.DataFrame, current_price: float, instrument: str = "NIFTY 50") -> Tuple[bool, str]:
        """
        Run all safety checks.
        Returns: (is_safe, reason)
        """
        
        # 1. Check if Breaker already tripped
        if self.triggered:
            elapsed = (datetime.now() - self.trigger_time).total_seconds() / 60
            if elapsed < self.pause_duration:
                return False, f"Circuit Breaker Active ({self.trigger_reason}) - {int(self.pause_duration - elapsed)}m remaining"
            else:
                self._reset_breaker()

        # 2. Flash Crash / Velocity Check (1-minute equivalent using last 5m candle limit)
        # Note: Ideally we check tick data or 1m data. Using 5m rapid move proxy.
        if not df_5m.empty:
            last_candle = df_5m.iloc[-1]
            high = last_candle['high']
            low = last_candle['low']
            open_p = last_candle['open']
            
            # Use High-Low range as proxy for volatility/velocity
            move_pct = ((high - low) / open_p) * 100
            
            # If a single 5m candle moves > 2x the 1min limit (approx), it's a crash/spike
            if move_pct > (MAX_1MIN_MOVE_PCT * 2.5): 
                self._trip_breaker("Flash Move Detected", 15)
                return False, f"Flash Crash Protection: {move_pct:.2f}% move in 5m"

        # 3. Expiry Day Gamma Guard
        if self._is_expiry_danger_zone(instrument):
             return False, "Expiry Gamma Guard Active (Post 2:00 PM)"

        # 4. Freak Trade Filter (Wick check)
        # If current price is far from last close but within candle? (Implied in Flash check)

        return True, "Market Normal"

    def _is_expiry_danger_zone(self, instrument: str) -> bool:
        """Check if it's Tuesday (Nifty Expiry) and past the Stop Time."""
        now = datetime.now()
        
        # NIFTY 50 Expiry = Tuesday (Weekday 1)
        # BANKNIFTY Expiry = Wednesday (Weekday 2) -> Future improvement
        
        is_expiry_day = False
        if "NIFTY" in instrument and "BANK" not in instrument:
             is_expiry_day = now.weekday() == 1  # 1 = Tuesday
        
        if not is_expiry_day:
            return False
            
        # Parse Stop Time
        stop_hour, stop_min = map(int, EXPIRY_STOP_TIME.split(":"))
        if now.time() >= time(stop_hour, stop_min):
            return True
            
        return False

    def _trip_breaker(self, reason: str, duration_mins: int):
        self.triggered = True
        self.trigger_reason = reason
        self.trigger_time = datetime.now()
        self.pause_duration = duration_mins
        logger.warning(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}. Pausing for {duration_mins} mins.")

    def _reset_breaker(self):
        self.triggered = False
        self.trigger_reason = None
        self.trigger_time = None
        logger.info("✅ Circuit Breaker Reset. Resuming operations.")
        
    def check_vix(self, vix_value: float) -> str:
        """Return market mode based on VIX"""
        if vix_value > VIX_PANIC_LEVEL:
            return "PANIC"
        if vix_value < VIX_LOW_LEVEL:
            return "DEAD"
        return "NORMAL"
