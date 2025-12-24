
import sys
import os
import logging
from datetime import datetime
import pandas as pd
from google.cloud import firestore

# Add project root to path
sys.path.append(os.getcwd())

from ml_module.model_storage import ModelStorage
from data_module.ml_data_collector import MLDataCollector
from retrain_function.main import prepare_data, train_quick_model
from config.settings import GOOGLE_CLOUD_PROJECT, ML_MODEL_BUCKET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_local_training():
    logger.info("🚀 Starting Local Training...")
    
    try:
        # 1. Connect to Firestore
        db = firestore.Client(project=GOOGLE_CLOUD_PROJECT)
        collector = MLDataCollector(db)
        
        # 2. Fetch Data
        logger.info("Fetching training data (direct dump)...")
        # Direct fetch to avoid index issues and field mismatches
        docs = db.collection("ml_training_data").stream()
        
        data = []
        for doc in docs:
            d = doc.to_dict()
            if d.get("label") is not None and "features" in d:
                data.append(d)
        
        if not data:
            logger.error("No training data found!")
            return
            
        logger.info(f"📊 Fetched {len(data)} training samples")
        
        # 3. Prepare Data
        X, y = prepare_data(data)
        
        # Convert categorical columns to 'category' dtype for LightGBM
        from ml_module.feature_extractor import get_categorical_features
        cat_features = get_categorical_features()
        for col in cat_features:
            if col in X.columns:
                X[col] = X[col].astype('category')
        
        # 4. Train Model
        logger.info("Training LightGBM model...")
        model, score = train_quick_model(X, y)
        
        logger.info(f"✅ Training Complete | Algo: LightGBM | Validation AUC: {score:.4f}")
        
        # 5. Save & Upload
        filename = f"model_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        model.save_model(filename)
        logger.info(f"💾 Model saved locally to {filename}")
        
        # Upload to GCS
        storage = ModelStorage(bucket_name=ML_MODEL_BUCKET)
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        if storage.upload_model(filename, version=version):
            logger.info(f"⬆️  Model uploaded to GCS bucket: {ML_MODEL_BUCKET}")
            logger.info(f"🎉 Deployment Ready! New model version: {version}")
        else:
            logger.error("❌ Failed to upload model to GCS")
            
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_local_training()
