"""
LightGBM Model Training Script
Train signal quality prediction model from Firestore data
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from google.cloud import firestore

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_module.feature_extractor import get_feature_names, get_categorical_features
from ml_module.model_storage import ModelStorage
from data_module.ml_data_collector import MLDataCollector
from config.settings import (
    ML_MODEL_BUCKET,
    ML_MIN_TRAINING_SAMPLES,
    GOOGLE_CLOUD_PROJECT
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_training_data(training_records: list) -> tuple:
    """
    Convert Firestore records to X, y format.
    
    Args:
        training_records: List of dicts from Firestore
        
    Returns:
        (X_df, y_series, metadata_df)
    """
    logger.info(f"Preparing {len(training_records)} training samples...")
    
    # Extract features and labels
    features_list = []
    labels = []
    metadata = []
    
    for record in training_records:
        features_list.append(record["features"])
        labels.append(record["label"])
        metadata.append({
            "signal_id": record["signal_id"],
            "timestamp": record["timestamp"],
            "outcome": record["outcome"]
        })
    
    # Create DataFrames
    X = pd.DataFrame(features_list)
    y = pd.Series(labels, name="label")
    meta = pd.DataFrame(metadata)
    
    logger.info(f"✅ Dataset prepared | Shape: {X.shape} | Win rate: {y.mean():.2%}")
    
    return X, y, meta


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> tuple:
    """
    Train LightGBM model with time-series cross-validation.
    
    Args:
        X: Features DataFrame
        y: Labels Series
        n_splits: Number of CV folds
        
    Returns:
        (model, cv_scores, feature_importance)
    """
    logger.info("🏋️ Training LightGBM model...")
    
    categorical_features = get_categorical_features()
    
    # LightGBM parameters
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,
        "verbose": -1
    }
    
    # Time-series cross-validation (important for time-series data!)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    
    logger.info(f"Running {n_splits}-fold time-series CV...")
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=categorical_features
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_data
        )
        
        # Train
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # Evaluate
        val_score = model.best_score["valid_0"]["auc"]
        cv_scores.append(val_score)
        
        logger.info(f"  Fold {fold}: AUC = {val_score:.4f}")
    
    avg_score = sum(cv_scores) / len(cv_scores)
    logger.info(f"✅ CV Complete | Avg AUC: {avg_score:.4f}")
    
    # Train final model on all data
    logger.info("Training final model on full dataset...")
    full_data = lgb.Dataset(
        X,
        label=y,
        categorical_feature=categorical_features
    )
    
    final_model = lgb.train(
        params,
        full_data,
        num_boost_round=150
    )
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": final_model.feature_importance()
    }).sort_values("importance", ascending=False)
    
    return final_model, cv_scores, importance


def main():
    """Main training pipeline."""
    logger.info("=" * 70)
    logger.info("🚀 LightGBM Model Training")
    logger.info("=" * 70)
    
    # 1. Connect to Firestore
    logger.info(f"Connecting to Firestore (Project: {GOOGLE_CLOUD_PROJECT})...")
    db = firestore.Client(project=GOOGLE_CLOUD_PROJECT)
    collector = MLDataCollector(db)
    
    # 2. Fetch training data
    logger.info("Fetching training data from last 90 days...")
    training_records = collector.get_training_data(
        days=90,
        min_samples=ML_MIN_TRAINING_SAMPLES
    )
    
    if len(training_records) < ML_MIN_TRAINING_SAMPLES:
        logger.error(
            f"❌ Insufficient training data: {len(training_records)} samples "
            f"(minimum: {ML_MIN_TRAINING_SAMPLES})"
        )
        logger.error("Run the agent for a few weeks to collect more data")
        sys.exit(1)
    
    # 3. Prepare data
    X, y, meta = prepare_training_data(training_records)
    
    # 4. Train model
    model, cv_scores, importance = train_model(X, y)
    
    # 5. Display results
    logger.info("\n" + "=" * 70)
    logger.info("📊 TRAINING RESULTS")
    logger.info("=" * 70)
    logger.info(f"Samples: {len(X)}")
    logger.info(f"Win Rate: {y.mean():.2%}")
    logger.info(f"CV AUC: {sum(cv_scores)/len(cv_scores):.4f} ± {pd.Series(cv_scores).std():.4f}")
    logger.info("\nTop 10 Features:")
    print(importance.head(10).to_string(index=False))
    
    # 6. Save model locally
    local_model_path = "models/signal_quality_v1.txt"
    os.makedirs("models", exist_ok=True)
    model.save_model(local_model_path)
    logger.info(f"\n✅ Model saved locally: {local_model_path}")
    
    # 7. Upload to GCS
    logger.info(f"\nUploading model to GCS bucket: {ML_MODEL_BUCKET}...")
    storage = ModelStorage(bucket_name=ML_MODEL_BUCKET)
    
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    if storage.upload_model(local_model_path, version=version):
        logger.info(f"✅ Model uploaded to GCS as version: {version}")
    else:
        logger.error("❌ Failed to upload model to GCS")
        sys.exit(1)
    
    logger.info("\n" + "=" * 70)
    logger.info("🎉 Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Next steps:")
    logger.info(f"  1. Set USE_ML_FILTERING=True in .env")
    logger.info(f"  2. Deploy updated Cloud Function")
    logger.info(f"  3. Monitor ML predictions in logs")


if __name__ == "__main__":
    main()
