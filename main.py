"""
NIFTY AI TRADING AGENT - Main Orchestrator
Coordinates: Data → Technical Analysis → AI → Telegram Alerts
"""

import logging
import sys
from datetime import datetime, time
from typing import Dict, List

import pytz

from config.settings import (
    INSTRUMENTS,
    ANALYSIS_START_TIME,
    MARKET_CLOSE_TIME,
    TIME_ZONE,
    DEBUG_MODE,
    MIN_SIGNAL_CONFIDENCE,
)

from data_module.fetcher import get_data_fetcher, DataFetcher
from analysis_module.technical import TechnicalAnalyzer
from ai_module.groq_analyzer import get_analyzer
from telegram_module.bot_handler import get_bot

from datetime import datetime, time


# ------------------------------------------------------
# Market Hours Guard (RUNS BEFORE ANY OTHER LOGIC)
# ------------------------------------------------------
def is_market_hours():
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).time()
    return time(9, 20) <= now <= time(15, 30)

if not is_market_hours():
    print("Outside market hours — exiting.")
    exit()
    
logger = logging.getLogger(__name__)


def setup_logging():
    log_format = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
    log_level = logging.DEBUG if DEBUG_MODE else logging.INFO

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("./logs/trading_agent.log", mode="a"),
        ],
    )

    return logging.getLogger(__name__)


logger = setup_logging()


class NiftyTradingAgent:
    """Main trading agent orchestrator."""

    def __init__(self):
        self.fetcher: DataFetcher = get_data_fetcher()
        self.ai_analyzer = get_analyzer()
        self.telegram_bot = get_bot()
        self.signals_generated: List[Dict] = []
        self.alerts_sent = 0

        logger.info("=" * 70)
        logger.info("🚀 NIFTY AI TRADING AGENT INITIALIZED")
        logger.info("=" * 70)
        logger.info(f"Mode: {'DEBUG' if DEBUG_MODE else 'PRODUCTION'}")
        logger.info(f"Timezone: {TIME_ZONE}")
        logger.info(f"Analysis start time: {ANALYSIS_START_TIME}")

    # =====================================================================
    # MAIN EXECUTION
    # =====================================================================

    def run_analysis(self, instruments: List[str] = None) -> Dict:
        """
        Main analysis execution loop.
        """
        if instruments is None:
            instruments = [
                k for k, v in INSTRUMENTS.items() if v.get("active", False)
            ]

        logger.info(f"\n📊 ANALYSIS STARTED | Instruments: {instruments}")
        logger.info("=" * 70)

        results = {
            "timestamp": datetime.now().isoformat(),
            "instruments_analyzed": 0,
            "signals_generated": 0,
            "alerts_sent": 0,
            "errors": 0,
            "details": {},
        }

        for instrument in instruments:
            try:
                logger.info(f"\n🔍 Analyzing: {instrument}")
                logger.info("-" * 70)

                instrument_result = self._analyze_single_instrument(instrument)
                results["details"][instrument] = instrument_result

                if instrument_result["success"]:
                    results["instruments_analyzed"] += 1
                    results["signals_generated"] += instrument_result[
                        "signals_count"
                    ]
                    results["alerts_sent"] += instrument_result["alerts_sent"]
                else:
                    results["errors"] += 1

            except Exception as e:
                logger.error(
                    f"❌ Error analyzing {instrument}: {str(e)}",
                    exc_info=DEBUG_MODE,
                )
                results["errors"] += 1

        self._print_analysis_summary(results)
        return results

    def _analyze_single_instrument(self, instrument: str) -> Dict:
        """Analyze single instrument with 5m + 15m data and MTF filters."""
        result = {
            "instrument": instrument,
            "success": False,
            "signals_count": 0,
            "alerts_sent": 0,
            "signals": [],
            "errors": [],
        }

        try:
            logger.debug("Step 1: Fetching real-time NSE data...")
            nse_data = self.fetcher.fetch_nse_data(instrument)

            if not nse:
                logger.error("❌ Failed to fetch NSE data")
                result["errors"].append("Data fetch failed")
                return result

            logger.debug("Step 2: Fetching 5m and 15m historical data...")
            df_5m = self.fetcher.fetch_historical_data(
                instrument, period="5d", interval="5m"
            )
            df_15m = self.fetcher.fetch_historical_data(
                instrument, period="10d", interval="15m"
            )

            if df_5m is None or df_5m.empty:
                logger.error("❌ No 5m historical data available")
                result["errors"].append("5m data unavailable")
                return result

            if df_15m is None or df_15m.empty:
                logger.error("❌ No 15m historical data available")
                result["errors"].append("15m data unavailable")
                return result

            df_5m = self.fetcher.preprocess_ohlcv(df_5m)
            df_15m = self.fetcher.preprocess_ohlcv(df_15m)

            logger.debug(
                f"5m shape: {df_5m.shape} | 15m shape: {df_15m.shape}"
            )

            analyzer = TechnicalAnalyzer(instrument)
            higher_tf_context = analyzer.get_higher_tf_context(df_15m)
            analysis = analyzer.analyze_with_multi_tf(
                df_5m, higher_tf_context
            )

            signals = self._generate_signals(instrument, analysis, nse_data)

            enriched_signals = []
            for sig in signals:
                if "BREAKOUT" in sig.get("signal_type", ""):
                    direction = (
                        "UP"
                        if "BULLISH" in sig["signal_type"]
                        else "DOWN"
                    )
                    is_false, fb_details = analyzer.detect_false_breakout(
                        df_5m, sig["price_level"], direction
                    )
                    sig["false_breakout"] = is_false
                    sig["false_breakout_details"] = fb_details
                enriched_signals.append(sig)

            final_signals = []
            for sig in enriched_signals:
                if sig.get("false_breakout"):
                    logger.info(
                        "⏭️  Suppressing alert due to false breakout | "
                        f"{sig['instrument']} @ {sig['price_level']:.2f}"
                    )
                    # Optionally send a dedicated false breakout alert:
                    # self.telegram_bot.send_false_breakout_alert(sig)
                    continue
                final_signals.append(sig)

            result["signals"] = final_signals
            result["signals_count"] = len(final_signals)

            for signal in final_signals:
                ai_analysis = self._get_ai_analysis(signal)
                signal["ai_analysis"] = ai_analysis

                if self._send_alert(signal):
                    result["alerts_sent"] += 1
                    self.alerts_sent += 1

            logger.info(
                f"✅ {instrument} analysis complete | "
                f"{result['signals_count']} signals | "
                f"{result['alerts_sent']} alerts sent"
            )
            result["success"] = True
            return result

        except Exception as e:
            logger.error(f"❌ Exception: {str(e)}", exc_info=DEBUG_MODE)
            result["errors"].append(str(e))
            return result

    def _generate_signals(
        self, instrument: str, analysis: Dict, nse_: Dict
    ) -> List[Dict]:
        """
        Generate trading signals from analysis with confidence gate.
        """
        signals: List[Dict] = []
        current_price = float(nse_data.get("price", 0) or 0)

        try:
            breakout = analysis.get("breakout_signal")
            retest = analysis.get("retest_signal")
            inside_bar = analysis.get("inside_bar_signal")

            if breakout:
                sig = {
                    "instrument": instrument,
                    "signal_type": breakout.signal_type.value,
                    "entry_price": breakout.entry_price,
                    "stop_loss": breakout.stop_loss,
                    "take_profit": breakout.take_profit,
                    "confidence": breakout.confidence,
                    "volume_confirmed": breakout.volume_confirmed,
                    "momentum_confirmed": breakout.momentum_confirmed,
                    "risk_reward_ratio": breakout.risk_reward_ratio,
                    "description": breakout.description,
                    "price_level": breakout.price_level,
                    "timestamp": breakout.timestamp.isoformat(),
                }
                if sig["confidence"] >= MIN_SIGNAL_CONFIDENCE:
                    logger.info(
                        f"   🚀 {sig['signal_type']} | Entry: {sig['entry_price']:.2f} "
                        f"| Conf: {sig['confidence']:.1f}%"
                    )
                    signals.append(sig)
                else:
                    logger.info(
                        f"   ⏭️  {sig['signal_type']} dropped (low confidence "
                        f"{sig['confidence']:.1f}%)"
                    )

            if retest:
                sig = {
                    "instrument": instrument,
                    "signal_type": retest.signal_type.value,
                    "entry_price": retest.entry_price,
                    "stop_loss": retest.stop_loss,
                    "take_profit": retest.take_profit,
                    "confidence": retest.confidence,
                    "price_level": retest.price_level,
                    "description": retest.description,
                    "timestamp": retest.timestamp.isoformat(),
                }
                if sig["confidence"] >= MIN_SIGNAL_CONFIDENCE:
                    logger.info(
                        f"   🎯 {sig['signal_type']} | Level: {sig['price_level']:.2f} "
                        f"| Conf: {sig['confidence']:.1f}%"
                    )
                    signals.append(sig)

            if inside_bar:
                sig = {
                    "instrument": instrument,
                    "signal_type": inside_bar.signal_type.value,
                    "entry_price": inside_bar.entry_price,
                    "stop_loss": inside_bar.stop_loss,
                    "take_profit": inside_bar.take_profit,
                    "confidence": inside_bar.confidence,
                    "description": inside_bar.description,
                    "timestamp": inside_bar.timestamp.isoformat(),
                }
                if sig["confidence"] >= MIN_SIGNAL_CONFIDENCE:
                    logger.info(
                        f"   📊 {sig['signal_type']} | Conf: {sig['confidence']:.1f}%"
                    )
                    signals.append(sig)

            if analysis.get("breakout_signal") and not analysis.get(
                "volume_confirmed"
            ):
                logger.warning("   ⚠️  Breakout without volume confirmation")

            return signals

        except Exception as e:
            logger.error(f"❌ Signal generation failed: {str(e)}")
            return []

    def _get_ai_analysis(self, signal: Dict) -> Dict:
        """Get AI analysis for a given signal."""
        try:
            signal_data = {
                "instrument": signal.get("instrument"),
                "signal_type": signal.get("signal_type"),
                "price_level": signal.get("price_level"),
                "entry": signal.get("entry_price"),
                "sl": signal.get("stop_loss"),
                "tp": signal.get("take_profit"),
                "technical_data": {
                    "volume_confirmed": signal.get("volume_confirmed"),
                    "momentum_confirmed": signal.get("momentum_confirmed"),
                    "rr_ratio": signal.get("risk_reward_ratio"),
                },
            }
            analysis = self.ai_analyzer.analyze_signal(signal_data)
            logger.debug(
                f"   AI Analysis: {analysis.get('recommendation')} | "
                f"Conf: {analysis.get('confidence')}%"
            )
            return analysis

        except Exception as e:
            logger.warning(f"⚠️  AI analysis failed: {str(e)}")
            return {"recommendation": "HOLD", "confidence": 50}

    def _send_alert(self, signal: Dict) -> bool:
        """Send Telegram alert for a signal."""
        try:
            stype = signal.get("signal_type", "")

            if "BREAKOUT" in stype or "BREAKDOWN" in stype:
                success = self.telegram_bot.send_breakout_alert(signal)
            elif "RETEST" in stype or "SUPPORT_BOUNCE" in stype or "RESISTANCE_BOUNCE" in stype:
                success = self.telegram_bot.send_retest_alert(signal)
            elif "INSIDE_BAR" in stype:
                success = self.telegram_bot.send_inside_bar_alert(signal)
            else:
                msg = (
                    f"{stype}\nEntry: {signal.get('entry_price')}\n"
                    f"{signal.get('description', '')}"
                )
                success = self.telegram_bot.send_message(msg)

            if success:
                logger.info("   ✅ Telegram alert sent")
            else:
                logger.warning("   ⚠️  Telegram alert failed")
            return success

        except Exception as e:
            logger.error(f"❌ Alert sending failed: {str(e)}")
            return False

    # =====================================================================
    # UTILITY
    # =====================================================================

    def _print_analysis_summary(self, results: Dict):
        logger.info("\n" + "=" * 70)
        logger.info("📊 ANALYSIS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Timestamp: {results['timestamp']}")
        logger.info(
            f"Instruments Analyzed: {results['instruments_analyzed']}"
        )
        logger.info(
            f"Total Signals Generated: {results['signals_generated']}"
        )
        logger.info(f"Alerts Sent: {results['alerts_sent']}")
        logger.info(f"Errors: {results['errors']}")
        logger.info("=" * 70 + "\n")

    def is_market_hours(self) -> bool:
        """Check if current time is within market hours (Mon-Fri, IST)."""
        tz = pytz.timezone(TIME_ZONE)
        now = datetime.now(tz)

        market_open = datetime.strptime(
            ANALYSIS_START_TIME, "%H:%M"
        ).time()
        market_close = datetime.strptime(
            MARKET_CLOSE_TIME, "%H:%M"
        ).time()

        if now.weekday() > 4:
            logger.debug("📅 Market closed (weekend)")
            return False

        if market_open <= now.time() <= market_close:
            return True

        logger.debug(f"⏰ Outside market hours ({now.time()})")
        return False

    def get_statistics(self) -> Dict:
        """Return simple statistics."""
        return {
            "signals_generated": len(self.signals_generated),
            "alerts_sent": self.alerts_sent,
            "ai_usage": self.ai_analyzer.get_usage_stats(),
            "bot_stats": self.telegram_bot.get_stats(),
        }


def main():
    """Main entry point for local run or cloud function."""
    logger.info("🚀 Agent starting...")

    agent = NiftyTradingAgent()

    logger.info("\n🔌 Testing external connections...")
    if not agent.ai_analyzer.test_connection():
        logger.error("❌ Groq API connection failed")
        agent.telegram_bot.send_error_notification(
            "Groq API connection failed"
        )
        return

    if not agent.telegram_bot.test_connection():
        logger.error("❌ Telegram connection failed")
        return

    logger.info("✅ All connections successful\n")

    if agent.is_market_hours():
        results = agent.run_analysis()
        agent.telegram_bot.send_startup_message()
    else:
        logger.info("⏰ Outside market hours - skipping analysis")
        agent.telegram_bot.send_message(
            "⏰ Outside market hours - analysis paused"
        )

    stats = agent.get_statistics()
    logger.info("\n📈 STATISTICS")
    logger.info(f"   Signals: {stats['signals_generated']}")
    logger.info(f"   Alerts Sent: {stats['alerts_sent']}")
    logger.info(
        f"   AI Usage: {stats['ai_usage']['tokens_used']} tokens used"
    )


def cloud_function_handler(request):
    """Entry point for Google Cloud Functions."""
    logger.info("☁️  Cloud Function triggered")
    main()
    return {"status": "success", "message": "Analysis completed"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nifty AI Trading Agent")
    parser.add_argument(
        "--once", action="store_true", help="Run analysis once"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test connections only"
    )

    args = parser.parse_args()

    if args.test:
        agent = NiftyTradingAgent()
        logger.info("Testing connections...")
        agent.ai_analyzer.test_connection()
        agent.telegram_bot.test_connection()
    else:
        main()
