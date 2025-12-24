"""
Signal Failure Detection Module
Tracks recent signals and alerts when they start failing
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pytz

logger = logging.getLogger(__name__)


class SignalFailureDetector:
    """
    Monitors recent signals and detects when they're failing.
    Sends alerts for early exit opportunities.
    """
    
    def __init__(self, persistence):
        """
        Args:
            persistence: Firestore persistence instance for storing tracked signals
        """
        self.persistence = persistence
        self.max_tracked_signals = 5  # Track last 5 signals per instrument
        
    def track_signal(self, signal: Dict, instrument: str) -> None:
        """
        Store a signal for failure tracking.
        
        Args:
            signal: Signal dict with entry_price, stop_loss, direction
            instrument: 'NIFTY' or 'BANKNIFTY'
        """
        try:
            signal_type = signal.get('signal_type', '')
            entry_price = signal.get('entry_price') or signal.get('price_level')
            stop_loss = signal.get('stop_loss')
            
            if not entry_price or not stop_loss:
                logger.debug(f"⏭️ Not tracking {signal_type} - missing entry/SL")
                return
            
            # Determine direction
            is_bullish = 'BULLISH' in signal_type or 'SUPPORT' in signal_type
            direction = 'LONG' if is_bullish else 'SHORT'
            
            # Create tracking record
            tracked_signal = {
                'instrument': instrument,
                'signal_type': signal_type,
                'direction': direction,
                'entry_price': float(entry_price),
                'stop_loss': float(stop_loss),
                'target': float(signal.get('take_profit', 0)) if signal.get('take_profit') else None,
                'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).isoformat(),
                'status': 'ACTIVE',
                'alerts_sent': 0  # Count failure alerts to avoid spam
            }
            
            # Store in Firestore
            collection = f'tracked_signals_{instrument.lower()}'
            doc_id = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{signal_type}"
            
            self.persistence.db.collection(collection).document(doc_id).set(tracked_signal)
            
            logger.info(f"📍 Tracking {direction} signal @ {entry_price:.2f} (SL: {stop_loss:.2f})")
            
            # Cleanup old signals (keep only last 5)
            self._cleanup_old_signals(instrument)
            
        except Exception as e:
            logger.error(f"Failed to track signal: {e}")
    
    def check_signal_health(self, instrument: str, current_price: float) -> List[Dict]:
        """
        Check all active tracked signals for failures.
        
        Args:
            instrument: 'NIFTY' or 'BANKNIFTY'
            current_price: Current market price
            
        Returns:
            List of failing signals that need alerts
        """
        failing_signals = []
        
        try:
            collection = f'tracked_signals_{instrument.lower()}'
            
            # Get active signals from last 30 minutes
            cutoff_time = datetime.now(pytz.timezone('Asia/Kolkata')) - timedelta(minutes=30)
            
            docs = self.persistence.db.collection(collection)\
                .where('status', '==', 'ACTIVE')\
                .stream()
            
            for doc in docs:
                signal = doc.to_dict()
                
                # Parse timestamp
                signal_time = datetime.fromisoformat(signal['timestamp'])
                
                # Skip old signals
                if signal_time < cutoff_time:
                    continue
                
                # Check if failing
                failure_info = self._check_if_failing(signal, current_price)
                
                if failure_info:
                    failure_info['doc_id'] = doc.id
                    failing_signals.append(failure_info)
            
            return failing_signals
            
        except Exception as e:
            logger.error(f"Failed to check signal health: {e}")
            return []
    
    def _check_if_failing(self, signal: Dict, current_price: float) -> Optional[Dict]:
        """
        Check if a single signal is failing.
        
        Returns:
            Dict with failure info if failing, None otherwise
        """
        direction = signal['direction']
        entry = signal['entry_price']
        stop_loss = signal['stop_loss']
        
        # Calculate distance to stop loss
        sl_distance_abs = abs(stop_loss - entry)
        
        # Calculate how far price has moved against us (in points)
        if direction == 'LONG':
            # For LONG: negative if price dropped
            price_move_pts = current_price - entry
            distance_to_sl_pts = current_price - stop_loss  # Negative if not hit
            
        else:  # SHORT
            # For SHORT: negative if price rose
            price_move_pts = entry - current_price
            distance_to_sl_pts = stop_loss - current_price  # Negative if not hit
        
        # Calculate as percentage of entry price (for logging)
        price_change_pct = (price_move_pts / entry) * 100
        
        # Trigger failure alert if:
        # 1. Price moved 30% of the way to stop loss (early warning)
        # OR
        # 2. Price moved 70% of the way to stop loss (urgent warning)
        
        percent_to_sl = abs(price_move_pts) / sl_distance_abs if sl_distance_abs > 0 else 0
        
        is_failing = False
        urgency = "NORMAL"
        
        if price_move_pts < 0:  # Price moved against us
            if percent_to_sl >= 0.7:  # 70% to SL
                is_failing = True
                urgency = "URGENT"
            elif percent_to_sl >= 0.3:  # 30% to SL
                is_failing = True
                urgency = "WARNING"
        
        if is_failing and signal.get('alerts_sent', 0) < 2:  # Max 2 failure alerts per signal
            minutes_ago = (datetime.now(pytz.timezone('Asia/Kolkata')) - 
                          datetime.fromisoformat(signal['timestamp'])).total_seconds() / 60
            
            return {
                'signal': signal,
                'current_price': current_price,
                'price_change_pct': price_change_pct,
                'percent_to_sl': percent_to_sl * 100,  # Convert to percentage
                'minutes_since_entry': minutes_ago,
                'urgency': urgency
            }
        
        return None
    
    def mark_signal_alerted(self, doc_id: str, instrument: str) -> None:
        """Mark that we've sent a failure alert for this signal."""
        try:
            from google.cloud import firestore
            
            collection = f'tracked_signals_{instrument.lower()}'
            doc_ref = self.persistence.db.collection(collection).document(doc_id)
            
            # Increment alert counter
            doc_ref.update({
                'alerts_sent': firestore.Increment(1)
            })
            
        except Exception as e:
            logger.error(f"Failed to mark signal as alerted: {e}")
    
    def _cleanup_old_signals(self, instrument: str) -> None:
        """Keep only last N signals, delete older ones."""
        try:
            from google.cloud import firestore
            
            collection = f'tracked_signals_{instrument.lower()}'
            
            # Get all signals sorted by timestamp
            docs = self.persistence.db.collection(collection)\
                .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                .stream()
            
            # Delete signals beyond max_tracked_signals
            for i, doc in enumerate(docs):
                if i >= self.max_tracked_signals:
                    doc.reference.delete()
                    logger.debug(f"🗑️ Cleaned up old signal: {doc.id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old signals: {e}")
