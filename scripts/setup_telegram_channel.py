
import os
import sys
import logging
import requests
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("TelegramSetup")

def setup_channel():
    print("===========================================")
    print("📢 TELEGRAM CHANNEL SETUP WIZARD")
    print("===========================================")
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN not found in .env")
        return

    print("\nℹ️  Prerequisites:")
    print("1. Create your Telegram Channel")
    print("2. Add your existing bot as an ADMINISTRATOR to that channel")
    print("3. Ensure 'Post Messages' permission is enabled for the bot")
    
    channel_input = input("\nEnter your Channel Username (e.g. @MyChannel) or ID (e.g. -100xxxx): ").strip()
    
    if not channel_input:
        print("❌ Input required.")
        return

    # Auto-fix: If user typed text without @, assuming it's a username (not numeric ID)
    if not channel_input.startswith("@") and not channel_input.startswith("-") and not channel_input.isdigit():
        print(f"⚠️  Assuming you meant a public username: adding '@' prefix -> @{channel_input}")
        channel_input = f"@{channel_input}"
        
    print(f"\n🧪 Testing access to {channel_input}...")
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": channel_input,
        "text": "✅ Cloud Agent Test: Verified channel access successfully!"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()
        
        if data.get("ok"):
            print("✅ SUCCESS! Test message sent to channel.")
            
            # Correct ID handling
            numeric_id = data['result']['chat']['id']
            print(f"📝 Numeric Channel ID: {numeric_id}")
            
            save = input("\nDo you want to save this to .env? (y/n): ").lower()
            if save == 'y':
                with open(".env", "a") as f:
                    # Remove any existing entry first (simple append is fine for now, env takes last)
                    f.write(f"\nTELEGRAM_CHANNEL_ID={numeric_id}\n")
                print("💾 Saved TELEGRAM_CHANNEL_ID to .env")
                print("\n🎉 Setup Complete! Run the bot and alerts will be forwarded.")
            
        else:
            print(f"❌ FAILED to send message.")
            print(f"Error: {data.get('description')}")
            print("\n💡 TROUBLESHOOTING:")
            print("1. Did you enter the **PUBLIC LINK** username (e.g. @NiftyAlerts)?")
            print("   - Just naming the channel 'NiftyAlerts' is not enough.")
            print("   - Go to Channel Info -> Edit -> Channel Type -> Public -> Create Link.")
            print("2. If it is a PRIVATE channel:")
            print("   - You must use the numeric ID (starts with -100).")
            print("   - Open https://web.telegram.org, click your channel.")
            print("   - Look at URL: https://web.telegram.org/a/#-100123456789")
            print("   - The ID is that number: -100123456789")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    setup_channel()
