
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from data_module.persistence import get_persistence

def check_state():
    print("Locked & Loaded: Inspecting Persistence State...")
    pm = get_persistence()
    
    if not pm.db:
        print("❌ Firestore not initialized.")
        return

    today_id = datetime.now().strftime("%Y-%m-%d")
    print(f"Stats Doc ID: {today_id}")

    try:
        doc_ref = pm.db.collection(pm.collection_name).document(today_id)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            print("\n--- DAILY STATS ---")
            for k, v in data.items():
                if k != "events": # events can be huge
                    print(f"{k}: {v}")
                else:
                    print(f"events: {list(v.keys()) if isinstance(v, dict) else 'Unknown structure'}")
            
            # Check specifically for flags
            print("\n--- FLAGS ---")
            print(f"Startup Message Sent: {data.get('startup_msg_sent', False)}")
            print(f"Market Context (ORB) Sent: {data.get('market_context_msg_sent', False)}")
            print(f"Daily Summary Sent: {data.get('daily_summary_msg_sent', False)}")
        else:
            print("⚠️ No stats document found for today.")
            
    except Exception as e:
        print(f"❌ Error fetching stats: {e}")

if __name__ == "__main__":
    check_state()
