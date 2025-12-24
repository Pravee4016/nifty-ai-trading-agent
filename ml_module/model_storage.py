"""
Model Storage Handler for Google Cloud Storage
Manages loading and saving LightGBM models from/to GCS
"""

import os
import logging
import tempfile
from typing import Optional
from google.cloud import storage
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelStorage:
    """Handle model storage and retrieval from Google Cloud Storage."""
    
    def __init__(self, bucket_name: str, model_name: str = "signal_quality_v1.txt"):
        """
        Initialize GCS model storage.
        
        Args:
            bucket_name: GCS bucket name (without gs:// prefix)
            model_name: Model filename in bucket
        """
        self.bucket_name = bucket_name
        self.model_name = model_name
        self.local_cache_dir = "/tmp/ml_models"  # Cloud Functions writable directory
        
        # Create cache directory
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(bucket_name)
            logger.info(f"✅ Connected to GCS bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to GCS: {e}")
            self.bucket = None
    
    def download_model(self, force_refresh: bool = False) -> Optional[str]:
        """
        Download model from GCS to local cache.
        
        Args:
            force_refresh: Force download even if cached
            
        Returns:
            Local path to model file, or None if failed
        """
        if not self.bucket:
            logger.error("GCS bucket not available")
            return None
        
        local_path = os.path.join(self.local_cache_dir, self.model_name)
        
        # Check cache
        if os.path.exists(local_path) and not force_refresh:
            logger.debug(f"Using cached model: {local_path}")
            return local_path
        
        try:
            blob = self.bucket.blob(f"models/{self.model_name}")
            
            if not blob.exists():
                logger.error(f"Model not found in GCS: models/{self.model_name}")
                return None
            
            # Download
            blob.download_to_filename(local_path)
            logger.info(f"✅ Downloaded model from GCS: {self.model_name}")
            
            return local_path
            
        except Exception as e:
            logger.error(f"❌ Failed to download model from GCS: {e}")
            return None
    
    def upload_model(self, local_model_path: str, version: Optional[str] = None) -> bool:
        """
        Upload trained model to GCS.
        
        Args:
            local_model_path: Path to local model file
            version: Optional version string (default: timestamp)
            
        Returns:
            True if successful
        """
        if not self.bucket:
            logger.error("GCS bucket not available")
            return False
        
        try:
            # Create versioned filename
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            versioned_name = f"signal_quality_{version}.txt"
            
            # Upload versioned file
            blob_versioned = self.bucket.blob(f"models/versions/{versioned_name}")
            blob_versioned.upload_from_filename(local_model_path)
            logger.info(f"✅ Uploaded versioned model: {versioned_name}")
            
            # Update active model (symlink equivalent)
            blob_active = self.bucket.blob(f"models/{self.model_name}")
            blob_active.upload_from_filename(local_model_path)
            logger.info(f"✅ Updated active model: {self.model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload model to GCS: {e}")
            return False
    
    def list_model_versions(self, limit: int = 10) -> list:
        """
        List available model versions in GCS.
        
        Args:
            limit: Max number of versions to return
            
        Returns:
            List of model version filenames
        """
        if not self.bucket:
            return []
        
        try:
            blobs = self.bucket.list_blobs(prefix="models/versions/", max_results=limit)
            versions = [blob.name.split("/")[-1] for blob in blobs]
            return sorted(versions, reverse=True)  # Newest first
            
        except Exception as e:
            logger.error(f"❌ Failed to list model versions: {e}")
            return []
