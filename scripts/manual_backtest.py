
import sys
import os
import logging
from datetime import datetime, time, timedelta
import pandas as pd
import pytz
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set Environment Variables for Backtest
os.environ["DEPLOYMENT_MODE"] = "LOCAL"
os.environ["SEND_TEST_ALERTS"] = "True"
os.environ["LOGLEVEL"] = "INFO"

from main import NiftyTradingAgent
from data_module.persistence import PersistenceManager
from config.settings import TIME_ZONE, INSTRUMENTS

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Backtest")

class MockPersistence(PersistenceManager):
    """Mocks PersistenceManager to avoid writing to real Firestore."""
    def __init__(self):
        self.stats = {}
        self.events = {}
        logger.info("🛡️ Mock Persistence Initialized (Safe Mode)")

    def increment_stat(self, stat_name: str, value: int = 1):
        self.stats[stat_name] = self.stats.get(stat_name, 0) + value
        logger.info(f"💾 [MOCK] Incremented {stat_name} to {self.stats[stat_name]}")

    def add_event(self, event_type: str, event_data: dict):
        if event_type not in self.events:
            self.events[event_type] = []
        self.events[event_type].append(event_data)
        logger.info(f"💾 [MOCK] Added event {event_type}")

    def get_daily_stats(self):
        return self.stats
    
    def save_recent_alerts(self, recent_alerts): pass
    def get_recent_alerts(self): return {}

def run_backtest():
    logger.info("🎬 Starting Optimized Backtest (Today's Data)")
    
    # 1. Initialize Agent with Mock Persistence
    with patch('main.get_persistence', return_value=MockPersistence()) as mock_persist, \
         patch('ai_module.groq_analyzer.GroqAnalyzer.analyze_signal') as mock_ai:
        
        # Configure Mock AI
        mock_ai.return_value = {
            "verdict": "BUY",
            "confidence": 85,
            "reasoning": "Backtest Mock Analysis: Strong signal detected.",
            "risks": ["Mock Risk"],
            "recommendation": "BUY", 
            "summary": "Mock Summary"
        }

        # MOCK TELEGRAM to prevent real alerts and capture them for summary
        with patch('telegram_module.bot_handler.TelegramBot') as MockBot:
            mock_bot_instance = MockBot.return_value
            captured_alerts = []

            def capture_alert(msg):
                captured_alerts.append(f"MSG: {msg}")
                return True
            
            def capture_breakout(signal):
                captured_alerts.append(f"BREAKOUT: {signal.get('symbol', 'NIFTY')} @ {signal.get('price_level')} ({signal.get('signal_type')})")
                return True

            def capture_retest(signal):
                captured_alerts.append(f"RETEST: {signal.get('instrument', 'NIFTY')} @ {signal.get('price_level')} ({signal.get('signal_type')})")
                return True

            mock_bot_instance.send_message.side_effect = capture_alert
            mock_bot_instance.send_breakout_alert.side_effect = capture_breakout
            mock_bot_instance.send_retest_alert.side_effect = capture_retest
            mock_bot_instance.send_inside_bar_alert.side_effect = capture_retest # Reuse logic
            
            agent = NiftyTradingAgent()
            agent.persistence = mock_persist.return_value
            agent.telegram_bot = mock_bot_instance # Force inject mock

        # 2. Fetch 'Real' Data for Today ONCE
        logger.info("📥 Fetching market data (1 Month)...")
        try:
            df_5m_full = agent.fetcher.fetch_historical_data("NIFTY", period="1mo", interval="5m")
            if df_5m_full is None or df_5m_full.empty:
                logger.error("❌ No data fetched. Market might be closed or API down.")
                return
            df_5m_full = agent.fetcher.preprocess_ohlcv(df_5m_full)
        except Exception as e:
            logger.error(f"❌ Initial fetch failed: {e}")
            return
            
        # last_date = df_5m_full.index[-1].date()
        target_date = datetime.now().date()
        logger.info(f"📅 Backtesting Date: {target_date}")

        # 3. Time Travel Loop
        tz = pytz.timezone(TIME_ZONE)
        # Full day simulation for Nov 26
        start_time = tz.localize(datetime.combine(target_date, time(9, 15))) 
        end_time = tz.localize(datetime.combine(target_date, time(15, 30)))
        
        current_sim_time = start_time
        
        # MOCK THE NETWORK CALLS TO PREVENT TIMEOUTS
        agent.fetcher.fetch_realtime_data = MagicMock()
        agent.option_fetcher.fetch_option_chain = MagicMock(return_value=None) # Skip options for speed
        
        while current_sim_time <= end_time:
            # Patch datetime in both main and technical module
            with patch('main.datetime') as mock_datetime_main, \
                 patch('analysis_module.technical.datetime') as mock_datetime_tech, \
                 patch('analysis_module.manipulation_guard.datetime') as mock_datetime_guard:
                
                # Configure Mock for Main
                mock_datetime_main.now.return_value = current_sim_time
                mock_datetime_main.strptime = datetime.strptime
                
                # Configure Mock for Technical (must allow .now(tz) calls)
                # handle datetime.now(tz) by returning sim_time converted to tz
                def mocked_now(tz=None):
                    if tz:
                        return current_sim_time.astimezone(tz)
                    return current_sim_time
                
                mock_datetime_tech.now.side_effect = mocked_now
                mock_datetime_tech.strptime = datetime.strptime

                # Configure Mock for Manipulation Guard
                mock_datetime_guard.now.side_effect = mocked_now
                mock_datetime_guard.strptime = datetime.strptime
                mock_datetime_guard.fromtimestamp = datetime.fromtimestamp
                
                print(f"\n⏳ Simulation Time: {current_sim_time.strftime('%H:%M')}")

                # 3a. Check Scheduled Messages
                # Mock previous day stats for startup/context
                if not hasattr(agent.fetcher, 'mock_prev_stats'):
                    agent.fetcher.get_previous_day_stats = MagicMock(return_value={
                        "pdc": 24000, "pdh": 24100, "pdl": 23900 # Dummy
                    })
                    agent.fetcher.get_opening_range_stats = MagicMock(return_value={
                        "high": 24050, "low": 24000, "range": 50
                    })
                    agent.fetcher.mock_prev_stats = True

                agent.check_scheduled_messages()
                
                # 3b. Run Analysis
                if time(9, 15) <= current_sim_time.time() <= time(15, 30):
                    current_slice = df_5m_full[df_5m_full.index <= current_sim_time]
                    
                    if not current_slice.empty:
                        # DEBUG: Check if we are actually moving
                        last_row_time = current_slice.index[-1]
                        print(f"DEBUG: SimTime={current_sim_time} | DataTime={last_row_time} | Price={current_slice.iloc[-1]['close']}")

                        # Mock historical data return
                        agent.fetcher.fetch_historical_data = MagicMock(return_value=current_slice)
                        
                        # Mock Live Price from last candle close
                        last_close = current_slice.iloc[-1]['close']
                        agent.fetcher.fetch_realtime_data.return_value = {
                            "price": last_close, 
                            "lastPrice": last_close,
                            "timestamp": current_sim_time.isoformat()
                        }
                        
                        agent.run_analysis()
                
                # 3c. Check Market Closed Alert
                if current_sim_time.time() >= time(15, 30):
                     from main import check_and_send_market_closed_alert
                     with patch('main.get_persistence', return_value=agent.persistence):
                         check_and_send_market_closed_alert()

            current_sim_time += timedelta(minutes=5)

    logger.info("✅ Backtest Complete")
    
    print("\n" + "="*50)
    print("📊 BACKTEST SUMMARY (TODAY)")
    print("="*50)
    if captured_alerts:
        print(f"Total Alerts: {len(captured_alerts)}")
        for i, alert in enumerate(captured_alerts, 1):
            print(f"{i}. {alert}")
    else:
        print("No alerts generated.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_backtest()
