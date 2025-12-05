import logging
import sys
from main import NiftyTradingAgent

# Configure logging to show details
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def verify_system():
    logger.info("🔍 Starting System Verification...")
    
    # Initialize Agent
    try:
        agent = NiftyTradingAgent()
        logger.info("✅ Agent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        return

    # 1. Test Data Fetching directly
    logger.info("\n📊 Testing Data Fetching...")
    try:
        # Test with NIFTY
        instrument = "NIFTY"
        
        # Real-time data
        nse_data = agent.fetcher.fetch_nse_data(instrument)
        if nse_data:
            logger.info(f"✅ Real-time Data (NIFTY): Price={nse_data.get('price')}")
        else:
            logger.error("❌ Failed to fetch real-time data")

        # Historical data
        hist_data = agent.fetcher.fetch_historical_data(instrument, period="5d", interval="5m")
        if hist_data is not None and not hist_data.empty:
            logger.info(f"✅ Historical Data (NIFTY): {len(hist_data)} candles fetched")
        else:
            logger.error("❌ Failed to fetch historical data")
            
    except Exception as e:
        logger.error(f"❌ Data fetching test failed: {e}")

    # 2. Run Full Analysis (Bypassing Market Hours)
    logger.info("\n🧠 Running Analysis (Bypassing Market Hours Check)...")
    try:
        # We call run_analysis directly. 
        # Note: Since market is closed, this analyzes the last available data.
        results = agent.run_analysis(instruments=["NIFTY", "BANKNIFTY"])
        
        logger.info("\n📝 Analysis Results:")
        logger.info(f"   Instruments Analyzed: {results['instruments_analyzed']}")
        logger.info(f"   Signals Generated: {results['signals_generated']}")
        logger.info(f"   Errors: {results['errors']}")
        
        if results['instruments_analyzed'] > 0:
            logger.info("✅ Analysis verification passed!")
        else:
            logger.warning("⚠️  Analysis ran but no instruments were successfully analyzed.")
            
    except Exception as e:
        logger.error(f"❌ Analysis run failed: {e}")

if __name__ == "__main__":
    verify_system()
