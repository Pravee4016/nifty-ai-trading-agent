
import sys
import os
sys.path.append('.')

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)

from google.cloud import firestore
from data_module.ml_data_collector import MLDataCollector
from config.settings import GOOGLE_CLOUD_PROJECT

def check_data():
    try:
        print(f"Connecting to Firestore project: {GOOGLE_CLOUD_PROJECT}")
        db = firestore.Client(project=GOOGLE_CLOUD_PROJECT)
        collector = MLDataCollector(db)
        
        stats = collector.get_stats()
        print("\n📊 ML Data Statistics:")
        print(f"Total Records: {stats.get('total_records', 0)}")
        print(f"Labeled Records: {stats.get('labeled_records', 0)}")
        print(f"Pending Labels: {stats.get('pending_labels', 0)}")
        
        return stats.get('labeled_records', 0)
    except Exception as e:
        print(f"❌ Error checking data: {e}")
        return 0

if __name__ == "__main__":
    check_data()
