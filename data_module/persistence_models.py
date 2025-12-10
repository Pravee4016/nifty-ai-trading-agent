"""
Persistence Models
Data classes for structured storage and alerting keys.
"""

from dataclasses import dataclass
from datetime import datetime
import pytz
from typing import Dict, Optional

@dataclass(frozen=True)
class AlertKey:
    """
    Structured key for duplicate alert detection.
    Frozen so it can be hashed and used as a dict key.
    """
    instrument: str
    signal_type: str
    level_ticks: int  # price normalized to ticks
    date: str         # 'YYYY-MM-DD'
    
    def __str__(self):
        return f"{self.instrument}|{self.signal_type}|{self.level_ticks}|{self.date}"

def build_alert_key(signal: Dict, tick_size: float = 0.05) -> AlertKey:
    """
    Factory to create an AlertKey from a signal dictionary.
    """
    # Use price_level (for SR/Breakout) or entry_price (for patterns)
    level = signal.get("price_level") or signal.get("entry_price") or 0.0
    
    # Normalize to ticks to avoid float rounding dupes
    if tick_size > 0:
        level_ticks = int(round(level / tick_size))
    else:
        level_ticks = int(level)
        
    # Use localized date
    ist = pytz.timezone("Asia/Kolkata")
    today = datetime.now(ist).strftime("%Y-%m-%d")
    
    return AlertKey(
        instrument=signal["instrument"],
        signal_type=signal["signal_type"],
        level_ticks=level_ticks,
        date=today,
    )
