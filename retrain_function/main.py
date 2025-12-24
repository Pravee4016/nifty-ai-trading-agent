"""
Cloud Function for Weekly Model Retraining
Triggered by Cloud Scheduler
"""

import logging
from datetime import datetime
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from google.cloud import firestore

# Import from deployed package
from ml_module.feature_extractor import get_feature_names, get_categorical_features
from ml_module.model_storage import ModelStorage
from data_module.ml_data_collector import MLDataCollector
from config.settings import (
    ML_MODEL_BUCKET,
    ML_MIN_TRAINING_SAMPLES,
    GOOGLE_CLOUD_PROJECT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retrain_model(request):
    """
    Cloud Function entry point for weekly retraining.
    
    Args:
        request: Flask request object (unused)
        
    Returns:
        Response tuple (message, status_code)
    """
    logger.info("=" * 70)
    logger.info("🔄 Weekly Model Retraining Started")
    logger.info("=" * 70)
    
    try:
        # 1. Connect to Firestore
        db = firestore.Client(project=GOOGLE_CLOUD_PROJECT)
        collector = MLDataCollector(db)
        
        # 2. Fetch training data
        logger.info("Fetching training data from last 90 days...")
        training_records = collector.get_training_data(
            days=90,
            min_samples=ML_MIN_TRAINING_SAMPLES
        )
        
        if len(training_records) < ML_MIN_TRAINING_SAMPLES:
            msg = (
                f"Insufficient data: {len(training_records)} samples "
                f"(min: {ML_MIN_TRAINING_SAMPLES})"
            )
            logger.warning(f"⚠️ {msg}")
            return (msg, 200)  # Not an error, just not enough data yet
        
        # 3. Prepare data
        X, y = prepare_data(training_records)
        
        # 4. Train model
        model, score = train_quick_model(X, y)
        
        # 5. Validate performance
        if score < 0.55:  # Must beat random (0.5)
            logger.warning(
                f"⚠️ Model underperforming (AUC: {score:.4f}), "
                "keeping previous model"
            )
            return (f"Model performance too low: {score:.4f}", 200)
        
        # 6. Save to GCS
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            model.save_model(tmp.name)
            tmp_path = tmp.name
        
        storage = ModelStorage(bucket_name=ML_MODEL_BUCKET)
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if storage.upload_model(tmp_path, version=version):
            logger.info(f"✅ Model retrained and uploaded | AUC: {score:.4f}")
            return (f"Success: Model v{version} | AUC: {score:.4f}", 200)
        else:
            logger.error("❌ Failed to upload model to GCS")
            return ("Failed to upload model", 500)
    
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}", exc_info=True)
        return (f"Error: {str(e)}", 500)


def prepare_data(training_records: list) -> tuple:
    """Prepare X, y from Firestore records."""
    features_list = [r["features"] for r in training_records]
    labels = [r["label"] for r in training_records]
    
    X = pd.DataFrame(features_list)
    y = pd.Series(labels)
    
    logger.info(f"Dataset: {X.shape} | Win rate: {y.mean():.2%}")
    return X, y


def train_quick_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Quick training with minimal CV for Cloud Function timeout.
    
    Returns:
        (model, validation_score)
    """
    categorical_features = get_categorical_features()
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbose": -1
    }
    
    # Single train/val split (last 20% for validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    score = model.best_score["valid_0"]["auc"]
    logger.info(f"Validation AUC: {score:.4f}")
    
    return model, score
