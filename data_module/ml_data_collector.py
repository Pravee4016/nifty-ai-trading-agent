"""
ML Data Collector
Collects and stores training data in Firestore
"""

import logging
from typing import Dict, Optional
from datetime import datetime
import pytz

from ml_module.feature_extractor import extract_features

logger = logging.getLogger(__name__)


class MLDataCollector:
    """Collect ML training data from live trading signals."""
    
    def __init__(self, firestore_client):
        """
        Initialize data collector.
        
        Args:
            firestore_client: Firestore client instance
        """
        self.db = firestore_client
        self.collection_name = "ml_training_data"
        
        logger.info("✅ ML Data Collector initialized")
    
    def record_signal(
        self,
        signal: Dict,
        technical_context: Dict,
        option_metrics: Dict,
        market_status: Dict = None
    ) -> Optional[str]:
        """
        Record a signal for ML training.
        
        Args:
            signal: Trading signal
            technical_context: MTF analysis context
            option_metrics: Options data
            market_status: Market conditions
            
        Returns:
            Document ID if successful
        """
        try:
            # Extract features
            features = extract_features(
                signal,
                technical_context,
                option_metrics,
                market_status
            )
            
            # Create training record
            ist = pytz.timezone("Asia/Kolkata")
            
            record = {
                "signal_id": signal.get("timestamp", datetime.now(ist).isoformat()),
                "instrument": signal.get("instrument"),
                "signal_type": signal.get("signal_type"),
                "features": features,
                "label": None,  # Will be updated when outcome is known
                "outcome": None,  # WIN/LOSS/PENDING
                "entry_price": signal.get("entry_price"),
                "stop_loss": signal.get("stop_loss"),
                "take_profit": signal.get("take_profit"),
                "timestamp": datetime.now(ist),
                "outcome_timestamp": None,
                "metadata": {
                    "confidence": signal.get("confidence"),
                    "score": signal.get("score"),
                    "ml_probability": signal.get("ml_probability")  # If ML was used
                }
            }
            
            # Store in Firestore
            doc_ref = self.db.collection(self.collection_name).document()
            doc_ref.set(record)
            
            logger.info(f"📝 ML training data recorded: {doc_ref.id}")
            return doc_ref.id
            
        except Exception as e:
            logger.error(f"❌ Failed to record ML training data: {e}")
            return None
    
    def update_outcome(
        self,
        signal_id: str,
        outcome: str,
        actual_exit_price: float = None
    ) -> bool:
        """
        Update signal outcome when trade closes.
        
        Args:
            signal_id: Document ID or signal timestamp
            outcome: "WIN" or "LOSS"
            actual_exit_price: Exit price if available
            
        Returns:
            True if successful
        """
        try:
            # Find document by signal_id
            docs = self.db.collection(self.collection_name).where(
                "signal_id", "==", signal_id
            ).limit(1).get()
            
            if not docs:
                logger.warning(f"Signal not found: {signal_id}")
                return False
            
            doc = docs[0]
            
            # Update with outcome
            ist = pytz.timezone("Asia/Kolkata")
            doc.reference.update({
                "outcome": outcome,
                "label": 1 if outcome == "WIN" else 0,
                "outcome_timestamp": datetime.now(ist),
                "actual_exit_price": actual_exit_price
            })
            
            logger.info(f"✅ Updated outcome for {signal_id}: {outcome}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update outcome: {e}")
            return False
    
    def get_training_data(
        self,
        days: int = 90,
        min_samples: int = 50
    ) -> list:
        """
        Fetch completed training samples from Firestore.
        
        Args:
            days: Number of days to look back
            min_samples: Minimum samples required
            
        Returns:
            List of training records with labels
        """
        try:
            from datetime import timedelta
            ist = pytz.timezone("Asia/Kolkata")
            cutoff = datetime.now(ist) - timedelta(days=days)
            
            # Query completed records (outcome != None)
            docs = self.db.collection(self.collection_name).where(
                "outcome_timestamp", ">=", cutoff
            ).where(
                "label", "!=", None
            ).get()
            
            training_data = []
            for doc in docs:
                data = doc.to_dict()
                if data.get("label") is not None:  # Ensure labeled
                    training_data.append(data)
            
            logger.info(f"📊 Fetched {len(training_data)} training samples")
            
            if len(training_data) < min_samples:
                logger.warning(
                    f"⚠️ Only {len(training_data)} samples available "
                    f"(min: {min_samples})"
                )
            
            return training_data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch training data: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get statistics on collected data."""
        try:
            # Total records
            total = len(self.db.collection(self.collection_name).get())
            
            # Labeled records
            labeled = len(
                self.db.collection(self.collection_name)
                .where("label", "!=", None)
                .get()
            )
            
            # Pending labels
            pending = total - labeled
            
            return {
                "total_records": total,
                "labeled_records": labeled,
                "pending_labels": pending,
                "collection": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {}
