"""
Persistence Module
Handles state storage using Google Cloud Firestore.
Required for stateless Cloud Function execution to track daily stats.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
import json
import pytz

from google.cloud import firestore
from config.settings import TIME_ZONE, DEBUG_MODE

logger = logging.getLogger(__name__)

class PersistenceManager:
    """Manages daily state in Firestore."""
    
    def __init__(self):
        self.db = None
        self.collection_name = "daily_stats"
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        
        if not self.project_id:
            logger.warning("⚠️ GOOGLE_CLOUD_PROJECT not set, persistence disabled")
            return

        try:
            self.db = firestore.Client(project=self.project_id)
            logger.info(f"💾 Firestore initialized | Project: {self.project_id}")
        except Exception as e:
            logger.error(f"❌ Firestore init failed: {str(e)}")

    def _get_today_doc_id(self) -> str:
        """Get document ID for today (YYYY-MM-DD)."""
        return datetime.now().strftime("%Y-%m-%d")

    def increment_stat(self, stat_name: str, value: int = 1):
        """Increment a counter stat."""
        if not self.db:
            return

        try:
            doc_ref = self.db.collection(self.collection_name).document(self._get_today_doc_id())
            
            # Use atomic increment
            doc_ref.set({
                stat_name: firestore.Increment(value),
                "last_updated": firestore.SERVER_TIMESTAMP
            }, merge=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to increment {stat_name}: {str(e)}")

    def _sanitize_for_firestore(self, data):
        """Recursively convert numpy types to standard Python types."""
        try:
            import numpy as np
            
            if isinstance(data, dict):
                return {k: self._sanitize_for_firestore(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [self._sanitize_for_firestore(v) for v in data]
            elif isinstance(data, (np.integer, np.int64, np.int32)):
                return int(data)
            elif isinstance(data, (np.floating, np.float64, np.float32)):
                return float(data)
            elif isinstance(data, np.bool_):
                return bool(data)
            elif hasattr(data, "item"):  # Generic numpy scalar check
                return data.item()
            return data
        except ImportError:
            return data
        except Exception as e:
            logger.warning(f"⚠️ Sanitization failed: {e}")
            return data

    def add_event(self, event_type: str, event_data: Dict):
        """Add an event (breakout, signal) to the daily list."""
        if not self.db:
            return

        try:
            # Sanitize data (convert numpy types to python types)
            clean_data = self._sanitize_for_firestore(event_data)
            
            doc_ref = self.db.collection(self.collection_name).document(self._get_today_doc_id())
            
            # Firestore array_union to append to list
            doc_ref.set({
                f"events.{event_type}": firestore.ArrayUnion([clean_data]),
                "last_updated": firestore.SERVER_TIMESTAMP
            }, merge=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to add event {event_type}: {str(e)}")

    def get_daily_stats(self) -> Dict:
        """Retrieve all stats for today."""
        if not self.db:
            return {}

        try:
            doc_ref = self.db.collection(self.collection_name).document(self._get_today_doc_id())
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            
            # If doc doesn't exist, create it initialized
            initial_stats = {"created_at": firestore.SERVER_TIMESTAMP}
            doc_ref.set(initial_stats, merge=True)
            return initial_stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get daily stats: {str(e)}")
            return {}
    
    
    def on_new_trading_day(self):
        """
        Reset daily stats and alerts if the date has changed.
        Should be called at startup.
        """
        if not self.db: return
        
        try:
            today_id = self._get_today_doc_id()
             # We can't easily "check" if it's a new day without storing "last_run_date"
             # But if we rely on _get_today_doc_id() for daily_stats, we just need to ensure 
             # recent_alerts are cleared if they belong to previous days.
            pass
        except Exception as e:
            logger.error(f"❌ Failed to handle new trading day: {e}")

    def save_recent_alerts(self, recent_alerts: Dict):
        """
        Save recent alerts to Firestore for duplicate detection across executions.
        
        Args:
            recent_alerts: Dict of {AlertKey: timestamp}
        """
        if not self.db:
            return
        
        try:
            doc_ref = self.db.collection("system_state").document("recent_alerts")
            
            # Convert AlertKey objects to string representation for JSON compatibility
            # Keys must be strings in Firestore maps
            alerts_serialized = {}
            for key, timestamp in recent_alerts.items():
                k_str = str(key) if not isinstance(key, str) else key
                alerts_serialized[k_str] = timestamp.isoformat()
            
            doc_ref.set({
                "alerts": alerts_serialized,
                "updated_at": firestore.SERVER_TIMESTAMP
            })
            
            logger.debug(f"💾 Saved {len(recent_alerts)} recent alerts to Firestore")
        
        except Exception as e:
            logger.error(f"❌ Failed to save recent alerts: {e}")
    
    def get_recent_alerts(self) -> Dict:
        """
        Retrieve recent alerts from Firestore.
        
        Returns:
            Dict of {AlertKey: timestamp}
        """
        from data_module.persistence_models import AlertKey
        
        if not self.db:
            return {}
        
        recent_alerts = {}
        try:
            doc_ref = self.db.collection("system_state").document("recent_alerts")
            doc = doc_ref.get()
            
            if not doc.exists:
                return {}
            
            data = doc.to_dict()
            alerts_serialized = data.get("alerts", {})
            
            ist = pytz.timezone("Asia/Kolkata")
            today = datetime.now(ist).strftime("%Y-%m-%d")
            
            for key_str, timestamp_str in alerts_serialized.items():
                try:
                    # Parse timestamp
                    dt = datetime.fromisoformat(timestamp_str)
                    if dt.tzinfo is None:
                        dt = ist.localize(dt)
                    
                    # Parse Key string back to AlertKey if possible
                    # Format: instrument|signal_type|level_ticks|date
                    parts = key_str.split("|")
                    if len(parts) == 4:
                        if parts[3] != today:
                            # Skip alerts from previous days
                            continue
                            
                        alert_key = AlertKey(
                            instrument=parts[0],
                            signal_type=parts[1],
                            level_ticks=int(parts[2]),
                            date=parts[3]
                        )
                        recent_alerts[alert_key] = dt
                    else:
                        # Legacy string key support (or if format changes)
                        # We might choose to drop legacy keys to force clean state
                        pass 
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse alert entry {key_str}: {e}")
            
            logger.info(f"📂 Loaded {len(recent_alerts)} active alerts for today from Firestore")
            return recent_alerts
        
        except Exception as e:
            logger.error(f"❌ Failed to get recent alerts: {e}")
            # Fail Closed: Return empty dict means we might re-alert if DB is down but memory is empty.
            # However, main.py will populate memory as it runs.
            return {}

# =========================================================================
# SINGLETON
# =========================================================================

_persistence: Optional[PersistenceManager] = None


def get_persistence() -> PersistenceManager:
    """Singleton pattern for PersistenceManager."""
    global _persistence
    if _persistence is None:
        _persistence = PersistenceManager()
    return _persistence
