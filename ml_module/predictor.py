"""
LightGBM Signal Quality Predictor
Optimized for Google Cloud Functions
"""

import logging
import lightgbm as lgb
from typing import Dict, Optional
import pandas as pd

from ml_module.model_storage import ModelStorage
from ml_module.feature_extractor import get_feature_names, get_categorical_features

logger = logging.getLogger(__name__)


class SignalQualityPredictor:
    """Predict signal quality using LightGBM model."""
    
    def __init__(self, bucket_name: str, model_name: str = "signal_quality_v1.txt"):
        """
        Initialize predictor with GCS model.
        
        Args:
            bucket_name: GCS bucket for models
            model_name: Model filename
        """
        self.model = None
        self.enabled = False
        self.feature_names = get_feature_names()
        self.categorical_features = get_categorical_features()
        
        # Initialize model storage
        self.storage = ModelStorage(bucket_name, model_name)
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load LightGBM model from GCS.
        
        Returns:
            True if successful
        """
        try:
            # Download model from GCS
            model_path = self.storage.download_model()
            
            if model_path is None:
                logger.warning("⚠️ Model file not available, ML filtering disabled")
                return False
            
            # Load LightGBM model
            self.model = lgb.Booster(model_file=model_path)
            self.enabled = True
            
            logger.info(f"✅ LightGBM Model Loaded | Features: {len(self.feature_names)}")
            return True
            
        except FileNotFoundError:
            logger.warning("⚠️ Model file not found, ML filtering disabled")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def predict_quality(self, features: Dict) -> Optional[float]:
        """
        Predict signal win probability.
        
        Args:
            features: Feature dict from feature_extractor
            
        Returns:
            Probability [0-1], or None if model disabled/error
        """
        if not self.enabled or self.model is None:
            return None
        
        try:
            # Convert features dict to DataFrame
            # LightGBM expects 2D array (even for single prediction)
            df = pd.DataFrame([features])
            
            # Ensure all expected features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    logger.warning(f"Missing feature: {feature}, using default")
                    df[feature] = 0 if feature not in self.categorical_features else "UNKNOWN"
            
            # Reorder columns to match training
            df = df[self.feature_names]
            
            # Predict (returns array)
            prediction = self.model.predict(df)[0]
            
            logger.debug(f"ML Prediction: {prediction:.3f}")
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}", exc_info=True)
            return None
    
    def predict_with_threshold(
        self,
        features: Dict,
        threshold: float = 0.65
    ) -> tuple[bool, float]:
        """
        Predict and return both decision and probability.
        
        Args:
            features: Feature dict
            threshold: Minimum probability to accept signal
            
        Returns:
            (should_accept, probability)
        """
        prob = self.predict_quality(features)
        
        if prob is None:
            # Fallback: accept signal if model unavailable
            return True, 0.5
        
        should_accept = prob >= threshold
        return should_accept, prob
    
    def reload_model(self) -> bool:
        """
        Force reload model from GCS (for updates).
        
        Returns:
            True if successful
        """
        logger.info("🔄 Reloading model from GCS...")
        return self._load_model()
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get feature importance from model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dict of {feature_name: importance_score}
        """
        if not self.enabled or self.model is None:
            return {}
        
        try:
            importance = self.model.feature_importance()
            feature_importance = dict(zip(self.feature_names, importance))
            
            # Sort and return top N
            sorted_importance = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return dict(sorted_importance[:top_n])
            
        except Exception as e:
            logger.error(f"❌ Failed to get feature importance: {e}")
            return {}
