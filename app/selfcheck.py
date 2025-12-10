"""
Self-Check Module
Verifies environment, configuration, and API connectivity.
"""
import logging
import os
import sys

# Ensure project root is in sys.path to allow imports from config/ and app/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import validate_config, LOG_DIR, CACHE_DIR
from config.logging_config import setup_logging
from app.bootstrap import create_agent

logger = setup_logging(__name__)

def check_directories():
    """Verify write access to critical directories."""
    dirs = [LOG_DIR, CACHE_DIR]
    success = True
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            test_file = os.path.join(d, "test_write.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"✅ Write access confirmed: {d}")
        except Exception as e:
            logger.error(f"❌ Write access FAILED: {d} | {e}")
            success = False
    return success

def run_self_check():
    """Run full system check."""
    logger.info("🔍 Starting System Self-Check...")
    
    # 1. Config Validation
    errors = validate_config()
    if errors:
        for e in errors:
            logger.error(e)
        logger.error("❌ Configuration Validation FAILED")
        return False
    logger.info("✅ Configuration Validated")
    
    # 2. Directory Access
    if not check_directories():
        return False

    # 3. Agent Initialization & Connectivity
    try:
        agent = create_agent()
        
        # Grid Connectivity
        groq_ok = agent.ai_analyzer.test_connection()
        if groq_ok:
            logger.info("✅ Groq API Connected")
        else:
            logger.error("❌ Groq API Connection FAILED")
            
        tg_ok = agent.telegram_bot.test_connection()
        if tg_ok:
            logger.info("✅ Telegram Bot Connected")
        else:
            logger.error("❌ Telegram Bot Connection FAILED")
            
        if groq_ok and tg_ok:
            logger.info("✅ All Systems Operational")
            return True
        else:
            return False
            
    except Exception as e:
        logger.critical(f"❌ Agent Bootstrap Failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_self_check()
    exit(0 if success else 1)
