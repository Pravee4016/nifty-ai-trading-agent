
import sys
import os
from google.cloud import firestore

# Force use of project from gcloud config if env var not set properly, or just use what's provided
project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "nifty-trading-agent")

print(f"🔍 Inspecting Firestore Project: {project_id}")

try:
    db = firestore.Client(project=project_id)
    collections = db.collections()
    
    print("\n📂 Collections found:")
    count = 0
    for collection in collections:
        print(f" - {collection.id}")
        
        # Check first few docs in each collection to see what they look like
        docs = list(collection.limit(3).stream())
        print(f"   (Contains approx {len(docs)}+ documents)")
        if docs:
            print(f"   Sample ID: {docs[0].id}")
            print(f"   Sample Keys: {list(docs[0].to_dict().keys())}")
        count += 1
        
    if count == 0:
        print("\n⚠️ No collections found in this project.")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
