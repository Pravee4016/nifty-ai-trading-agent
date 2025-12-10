"""
NIFTY AI TRADING AGENT - Main Orchestrator
Coordinates: Data → Technical Analysis → AI → Telegram Alerts
"""

import sys
import os
import logging
from datetime import datetime, time, timedelta
import pytz
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from config.settings import (
    INSTRUMENTS,
    ANALYSIS_START_TIME,
    MARKET_CLOSE_TIME,
    TIME_ZONE,
    DEBUG_MODE,
    MIN_VOLUME_RATIO,
    MIN_SIGNAL_CONFIDENCE,
)
# MIN_SIGNAL_CONFIDENCE is imported from settings

from data_module.fetcher import get_data_fetcher, DataFetcher
from analysis_module.signal_pipeline import SignalPipeline
from analysis_module.technical import TechnicalAnalyzer, Signal, TechnicalLevels
from ai_module.groq_analyzer import get_analyzer, GroqAnalyzer
from telegram_module.bot_handler import get_bot, TelegramBotHandler, format_signal_message
from data_module.persistence import get_persistence
from data_module.persistence_models import AlertKey, build_alert_key
from data_module.trade_tracker import get_trade_tracker, TradeTracker
from data_module.option_chain_fetcher import OptionChainFetcher

from analysis_module.option_chain_analyzer import OptionChainAnalyzer
from analysis_module.manipulation_guard import CircuitBreaker
from config.logging_config import setup_logging

# ------------------------------------------------------
# Market Hours Check (moved to cloud_function_handler)
# ------------------------------------------------------
def _is_market_hours_quick():
    """Quick market hours check for early exit."""
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).time()
    today = datetime.now(ist).weekday()
    # Allow execution until 16:30 for EOD activities (Summary, Market Closed Alert)
    # Weekdays only (0-4)
    if today > 4: return False
    return time(9, 15) <= now <= time(16, 30)

logger = setup_logging(__name__)


class NiftyTradingAgent:
    """Main trading agent orchestrator."""

    def __init__(self):
        self.fetcher: DataFetcher = get_data_fetcher()
        self.option_analyzer = OptionChainAnalyzer()
        self.trade_tracker = TradeTracker()
        self.bot_handler = TelegramBotHandler()
        self.telegram_bot = self.bot_handler  # Alias for consistency
        self.groq_analyzer = GroqAnalyzer()
        self.ai_analyzer = self.groq_analyzer # Alias for compatibility
        self.persistence = get_persistence()
        self.option_fetcher = OptionChainFetcher()
        self.circuit_breaker = CircuitBreaker()
        
        # Initialize Signal Pipeline
        self.signal_pipeline = SignalPipeline(groq_analyzer=self.groq_analyzer)
        
        self.signals_generated: List[Dict] = []
        self.alerts_sent = 0
        
        # Daily event tracking for EOD summary
        self.daily_breakouts: List[Dict] = []
        self.daily_breakdowns: List[Dict] = []
        self.daily_retests: List[Dict] = []
        self.daily_reversals: List[Dict] = []
        self.daily_data_fetches = 0
        self.daily_analyses = 0
        
        # Duplicate alert prevention - Load from Firestore (persistent across executions)
        self.recent_alerts: Dict[str, datetime] = self.persistence.get_recent_alerts()
        logger.info(f"📂 Loaded {len(self.recent_alerts)} recent alerts from Firestore")
        
        # Level-based memory (tracks S/R levels to prevent repeats all day)
        self.daily_level_memory: set = set()
        
        # Market Context for AI Analysis
        self.market_context: Dict = {}

        logger.info("=" * 70)
        logger.info("🤖 Nifty AI Trading Agent Initialized (Phase 3: AI Analyst)")
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

        # Track data fetch count
        self.persistence.increment_stat("data_fetches", len(instruments))

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
                
                self.persistence.increment_stat("analyses_run")

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
            logger.debug("Step 1: Fetching real-time market data (Fyers/YF)...")
            market_data = self.fetcher.fetch_realtime_data(instrument)

            if not market_data:
                logger.warning("⚠️  Market data fetch failed, using empty fallback")
                market_data = {}  # Empty dict as fallback


            logger.debug("Step 2: Fetching 5m and 15m historical data...")
            df_5m = self.fetcher.fetch_historical_data(
                instrument, period="5d", interval="5m"
            )
            df_15m = self.fetcher.fetch_historical_data(
                instrument, period="10d", interval="15m"
            )
            # Fetch daily data for previous day trend
            df_daily = self.fetcher.fetch_historical_data(
                instrument, period="5d", interval="1d"
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
            if df_daily is not None and not df_daily.empty:
                df_daily = self.fetcher.preprocess_ohlcv(df_daily)

            logger.debug(
                f"5m shape: {df_5m.shape} | 15m shape: {df_15m.shape}"
            )

            # --- MANIPULATION DEFENSE (Phase 4) ---
            current_price = market_data.get("price", df_5m.iloc[-1]['close'])
            is_safe, safety_reason = self.circuit_breaker.check_market_integrity(df_5m, current_price, instrument)
            if not is_safe:
                logger.warning(f"🛡️ MANIPULATION GUARD ACTIVE: {safety_reason}")
                result["errors"].append(f"SAFETY LOCK: {safety_reason}")
                # Log event but gracefully exit analysis for this instrument
                return result
            # --------------------------------------

            analyzer = TechnicalAnalyzer(instrument)
            higher_tf_context = analyzer.get_higher_tf_context(df_15m, df_5m, df_daily)
            
            # Update global market context for AI
            self.market_context[instrument] = {
                "trend_5m": higher_tf_context.get("trend_5m", "NEUTRAL"),
                "trend_15m": higher_tf_context.get("trend_15m", "NEUTRAL"),
                "trend_daily": higher_tf_context.get("trend_daily", "NEUTRAL"),
                "volatility_score": higher_tf_context.get("volatility_score", 0),
                "pdh": higher_tf_context.get("pdh", 0),
                "pdl": higher_tf_context.get("pdl", 0),
                "last_price": current_price
            }
            
            analysis = analyzer.analyze_with_multi_tf(
                df_5m, higher_tf_context, df_15m=df_15m
            )
            # Inject context for signal generation
            analysis["higher_tf_context"] = higher_tf_context
            
            # Check and auto-close open trades based on current price
            current_price = market_data.get("lastPrice", 0)
            if current_price > 0:
                closed = self.trade_tracker.check_open_trades({instrument: current_price})
                if closed > 0:
                    logger.info(f"   ✅ Auto-closed {closed} trade(s) for {instrument}")

            signals = self._generate_signals(instrument, analysis, market_data, df_5m)

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
                
                # Track events for daily summary
                self._track_daily_event(signal)

                if self._send_alert(signal):
                    result["alerts_sent"] += 1
                    self.alerts_sent += 1
            
            self.daily_analyses += 1

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
        self, instrument: str, analysis: Dict, nse_data: Dict, df_5m: pd.DataFrame = None
    ) -> List[Dict]:
        """
        Generate trading signals using SignalPipeline.
        """
        signals: List[Dict] = []
        
        # 1. Fetch Option Data (Inputs for Pipeline)
        option_metrics = {}
        try:
            logger.info(f"🧬 Fetching Option Chain for {instrument}...")
            oc_data = self.option_fetcher.fetch_option_chain(instrument)
            if oc_data:
                pcr_value = self.option_analyzer.calculate_pcr(oc_data)
                spot_price = float(nse_data.get("price", 0) or 0)
                iv_value = self.option_analyzer.calculate_atm_iv(oc_data, spot_price)
                oi_change_data = self.option_analyzer.analyze_oi_change(oc_data, spot_price)
                max_pain = self.option_analyzer.calculate_max_pain(oc_data)
                
                if pcr_value is not None: option_metrics["pcr"] = pcr_value
                if iv_value is not None: option_metrics["iv"] = iv_value
                if oi_change_data: option_metrics["oi_change"] = oi_change_data
                if max_pain is not None: option_metrics["max_pain"] = max_pain
                    
                logger.info(f"📊 Option Metrics: PCR={pcr_value}, IV={iv_value}%, MaxPain={max_pain}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate option metrics: {e}")

        # 2. Gather Raw Signals from Technical Analysis
        raw_signals = []
        
        # Extract signals from analysis object
        breakout = analysis.get("breakout_signal")
        retest = analysis.get("retest_signal")
        inside_bar = analysis.get("inside_bar_signal")
        pin_bar = analysis.get("pin_bar_signal")
        engulfing = analysis.get("engulfing_signal")
        
        for potential_signal in [breakout, retest, inside_bar, pin_bar, engulfing]:
            if not potential_signal:
                continue
                
            sig = {
                "instrument": instrument,
                "signal_type": potential_signal.signal_type.value,
                "entry_price": potential_signal.entry_price,
                "stop_loss": potential_signal.stop_loss,
                "take_profit": potential_signal.take_profit,
                "confidence": potential_signal.confidence, 
                "volume_confirmed": getattr(potential_signal, "volume_confirmed", False),
                "momentum_confirmed": getattr(potential_signal, "momentum_confirmed", True),
                "risk_reward_ratio": getattr(potential_signal, "risk_reward_ratio", 0),
                "description": potential_signal.description,
                "price_level": potential_signal.price_level,
                "timestamp": potential_signal.timestamp.isoformat(),
            }
            # Add simple technical gate here if needed, but Pipeline handles scoring
            if sig["confidence"] >= MIN_SIGNAL_CONFIDENCE:
                 raw_signals.append(sig)

        # 3. Market Status for Pipeline
        market_status = {}
        # Choppy check
        analyzer = TechnicalAnalyzer(instrument)
        choppy_df = df_5m if df_5m is not None else self.fetcher.get_historical_data(instrument, "5m", 100)
        is_choppy, choppy_reason = analyzer._is_choppy_session(choppy_df)
        market_status["is_choppy"] = is_choppy
        market_status["choppy_reason"] = choppy_reason

        # 4. Delegate to Pipeline
        try:
            processed_signals = self.signal_pipeline.process_signals(
                raw_signals=raw_signals,
                instrument=instrument,
                technical_context=analysis,
                option_metrics=option_metrics,
                recent_alerts=self.recent_alerts,
                market_status=market_status
            )
            return processed_signals
            
        except Exception as e:
            logger.error(f"❌ Signal Pipeline Error: {e}", exc_info=True)
            return []

    def _get_ai_analysis(self, signal: Dict) -> Dict:
        """Deprecated: AI is now called inside SignalPipeline."""
        return signal.get("ai_analysis", {})

    def _check_alert_limits(self, signal: Dict) -> Tuple[bool, str]:
        """
        Check if sending this alert would exceed daily limits.
        Uses AlertKey structure from recent_alerts.
        
        Returns:
            (can_send, rejection_reason)
        """
        try:
            from config.settings import (
                MAX_ALERTS_PER_DAY,
                MAX_ALERTS_PER_TYPE,
                MAX_ALERTS_PER_INSTRUMENT,
            )
            
            # Check total alerts from list size
            total_alerts = len(self.recent_alerts)
            if total_alerts >= MAX_ALERTS_PER_DAY:
                return False, f"Daily limit reached ({total_alerts}/{MAX_ALERTS_PER_DAY})"

            instrument = signal.get("instrument", "")
            signal_type = signal.get("signal_type", "")
            
            # Count per type
            recent_of_type = sum(
                1 for key in self.recent_alerts.keys()
                if hasattr(key, 'signal_type') and signal_type in key.signal_type
            )
            
            if recent_of_type >= MAX_ALERTS_PER_TYPE:
                return False, f"{signal_type} limit reached ({recent_of_type}/{MAX_ALERTS_PER_TYPE})"
            
            # Count per instrument
            recent_for_instrument = sum(
                1 for key in self.recent_alerts.keys()
                if hasattr(key, 'instrument') and key.instrument == instrument
            )
            
            if recent_for_instrument >= MAX_ALERTS_PER_INSTRUMENT:
                return False, f"{instrument} limit reached ({recent_for_instrument}/{MAX_ALERTS_PER_INSTRUMENT})"
            
            return True, ""
            
        except Exception as e:
            logger.warning(f"⚠️ Alert limit check failed, entering SAFE mode: {e}")
            # Fail Closed to prevent spam in degraded state
            return False, "ALERT_LIMIT_CHECK_FAILED"

    def _should_suppress_retest(self, signal: Dict) -> Tuple[bool, str]:
        """
        Specialized filter for RETEST alerts to reduce noise.
        
        Rules:
        1. Proximity: Block if within 0.2% of recent alert at same level (60m window).
        2. Conflict: Block if opposing retest at same level within 30m.
        """
        try:
            stype = signal.get("signal_type", "")
            if "RETEST" not in stype and "BOUNCE" not in stype:
                return False, ""  # Not a retest, don't suppress
            
            instrument = signal.get("instrument", "")
            price = float(signal.get("price_level", 0))
            current_direction = "LONG" if "BULLISH" in stype or "SUPPORT" in stype else "SHORT"
            
            ist = pytz.timezone("Asia/Kolkata")
            now = datetime.now(ist)
            
            for key, timestamp in self.recent_alerts.items():
                # Ensure key is AlertKey
                if not hasattr(key, "instrument"): continue
                
                if key.instrument != instrument:
                    continue
                
                # Retrieve approximate price from ticks
                # Assume 0.05 tick size (standard for Nifty indices)
                prev_price = key.level_ticks * 0.05
                
                # Check 1: Proximity (Same Level)
                # If within 0.2% price difference
                if abs(prev_price - price) < (price * 0.002):
                    prev_direction = "LONG" if "BULLISH" in key.signal_type or "SUPPORT" in key.signal_type else "SHORT"
                    time_diff = (now - timestamp).total_seconds() / 60.0
                    
                    # A. Same Direction (Duplicate) -> 60 min cooldown for Retests
                    if current_direction == prev_direction:
                        if time_diff < 60:
                            return True, f"Recent similar retest {time_diff:.1f}m ago"
                    
                    # B. Opposing Direction (Conflict) -> 30 min suppression
                    else:
                        if time_diff < 30:
                            return True, f"Conflicting retest ({prev_direction}) {time_diff:.1f}m ago"
                            
            return False, ""
            
        except Exception as e:
            logger.error(f"❌ Retest suppression check failed: {e}")
            return False, ""

    def _send_alert(self, signal: Dict) -> bool:
        """Send Telegram alert with structured duplicate prevention."""
        try:
            
            stype = signal.get("signal_type", "")
            instrument = signal.get("instrument", "")
            price_level = signal.get("price_level", 0)
            
            # Limit Check
            can_send, reject_reason = self._check_alert_limits(signal)
            if not can_send:
                logger.warning(f"⏭️ Alert limit: {reject_reason}")
                return False
            
            # Retest Filter
            if "RETEST" in stype or "BOUNCE" in stype:
                should_suppress, reason = self._should_suppress_retest(signal)
                if should_suppress:
                    logger.info(f"⏭️ Suppressing RETEST alert: {reason}")
                    return False
            
            # Structured Duplicate Check using AlertKey
            new_key = build_alert_key(signal)
            ist = pytz.timezone("Asia/Kolkata")
            now = datetime.now(ist)
            
            # 1. Exact/Zone Duplicate Check
            # Check if we have an alert with same key (same inst, type, level, date)
            # or very close level
            for key, timestamp in self.recent_alerts.items():
                if not hasattr(key, "instrument"): continue # skip legacy string keys if any
                
                if key.instrument == new_key.instrument and key.signal_type == new_key.signal_type:
                    # Check level proximity (within 3 ticks = 0.15 pts roughly)
                    # Actually let's use the explicit logic from before (0.1%)
                    # Convert ticks back to price for comparison
                    prev_price = key.level_ticks * 0.05
                    curr_price = new_key.level_ticks * 0.05
                    
                    if abs(prev_price - curr_price) < (curr_price * 0.001):
                        time_diff = (now - timestamp).total_seconds() / 60.0
                        if time_diff < 30: # 30 min cooldown
                            logger.info(f"⏭️ Duplicate Alert {time_diff:.1f}m ago | {key}")
                            return False
            
            # 2. Directional Conflict Check
            # Prevent LONG signal if SHORT sent recently at same level (and vice versa)
            current_direction = "LONG" if "BULLISH" in stype or "SUPPORT" in stype else "SHORT"
            
            for key, timestamp in self.recent_alerts.items():
                if not hasattr(key, "instrument"): continue

                if key.instrument == instrument:
                    prev_price = key.level_ticks * 0.05
                    # Check if nearby level (within 0.2%)
                    if abs(prev_price - price_level) < (price_level * 0.002):
                        prev_direction = "LONG" if "BULLISH" in key.signal_type or "SUPPORT" in key.signal_type else "SHORT"
                        
                        # If directions oppose and within 15 mins
                        if current_direction != prev_direction:
                            conflict_diff = (now - timestamp).total_seconds() / 60.0
                            if conflict_diff < 15:
                                logger.info(
                                    f"⏭️ Skipping conflicting signal | {current_direction} vs recent {prev_direction} | "
                                    f"Diff: {conflict_diff:.1f} mins"
                                )
                                return False
            
            # ====================
            # Send Alert
            # ====================
            if "BREAKOUT" in stype or "BREAKDOWN" in stype:
                success = self.telegram_bot.send_breakout_alert(signal)
            elif "RETEST" in stype or "SUPPORT_BOUNCE" in stype or "RESISTANCE_BOUNCE" in stype:
                success = self.telegram_bot.send_retest_alert(signal)
            elif "INSIDE_BAR" in stype:
                success = self.telegram_bot.send_inside_bar_alert(signal)
            elif "PIN_BAR" in stype or "ENGULFING" in stype:
                # Use retest alert format for pin bars and engulfing (similar structure)
                success = self.telegram_bot.send_retest_alert(signal)
            else:
                msg = (
                    f"{stype}\nEntry: {signal.get('entry_price')}\n"
                    f"{signal.get('description', '')}"
                )
                success = self.telegram_bot.send_message(msg)

            if success:
                logger.info("   ✅ Telegram alert sent")
                self.persistence.increment_stat("alerts_sent")
                
                # Record trade for performance tracking
                trade_id = self.trade_tracker.record_alert(signal)
                if trade_id:
                    logger.info(f"   📝 Trade tracked: {trade_id}")
                
                # Record this alert to prevent duplicates
                self.recent_alerts[new_key] = now
                
                # Record level for all-day blocking
                level_key = f"{instrument}_{stype}_level_{round(price_level / 25) * 25:.0f}"
                self.daily_level_memory.add(level_key)
                
                # Cleanup old entries (older than 6 hours)
                cutoff_time = now - timedelta(hours=6)
                self.recent_alerts = {
                    k: v for k, v in self.recent_alerts.items() 
                    if v > cutoff_time
                }
                
                # Save to Firestore for persistence across executions
                self.persistence.save_recent_alerts(self.recent_alerts)
            else:
                logger.warning("   ⚠️  Telegram alert failed")
            return success

        except Exception as e:
            logger.error(f"❌ Alert sending failed: {str(e)}")
            return False

    # =====================================================================
    # DAILY EVENT TRACKING & SUMMARY
    # =====================================================================

    def _track_daily_event(self, signal: Dict):
        """Track signal event for daily summary."""
        signal_type = signal.get("signal_type", "")
        
        # Local tracking - categorize all signal types
        if signal_type == "BULLISH_BREAKOUT":
            self.daily_breakouts.append(signal)
            self.persistence.add_event("breakouts", signal)
        elif signal_type == "BEARISH_BREAKOUT" or "BREAKDOWN" in signal_type:
            self.daily_breakdowns.append(signal)
            self.persistence.add_event("breakdowns", signal)
        elif any(x in signal_type for x in ["RETEST", "BOUNCE", "PIN_BAR", "ENGULFING", "INSIDE_BAR"]):
            self.daily_retests.append(signal)
            self.persistence.add_event("retests", signal)
        # Could add reversal detection logic here if needed

    def generate_daily_summary(self) -> Dict:
        """Generate comprehensive end-of-day market summary."""
        try:
            logger.info("📊 Generating end-of-day market summary")
            
            # Fetch persisted stats
            stored_stats = self.persistence.get_daily_stats()
            events = stored_stats.get("events", {})
            
            instruments = [
                k for k, v in INSTRUMENTS.items() if v.get("active", False)
            ]
            
            summary = {
                "timestamp": datetime.now().isoformat(),
                "instruments": {},
                "statistics": {
                    "data_fetches": stored_stats.get("data_fetches", 0),
                    "analyses_run": stored_stats.get("analyses_run", 0),
                    "alerts_sent": stored_stats.get("alerts_sent", 0),
                    "breakouts": len(events.get("breakouts", [])),
                    "breakdowns": len(events.get("breakdowns", [])),
                    "retests": len(events.get("retests", [])),
                    "reversals": len(events.get("reversals", [])),
                },
                "events": events
            }
            
            # Get latest data for each instrument
            for instrument in instruments:
                try:
                    # Fetch latest daily candle
                    df_day = self.fetcher.fetch_historical_data(
                        instrument, period="2d", interval="1d"
                    )
                    
                    if df_day is not None and not df_day.empty:
                        df_day = self.fetcher.preprocess_ohlcv(df_day)
                        latest = df_day.iloc[-1]
                        
                        # Get PDH/PDL for context
                        pdh_pdl = self.fetcher.get_previous_day_stats(instrument)
                        
                        # Get intraday data for trend analysis
                        df_5m = self.fetcher.fetch_historical_data(
                            instrument, period="1d", interval="5m"
                        )
                        
                        # Calculate short-term trend (last hour of trading)
                        short_term_trend = "NEUTRAL"
                        long_term_trend = "NEUTRAL"
                        
                        if df_5m is not None and not df_5m.empty and len(df_5m) >= 12:
                            df_5m = self.fetcher.preprocess_ohlcv(df_5m)
                            last_12 = df_5m.tail(12)  # Last hour
                            
                            if last_12.iloc[-1]["close"] > last_12.iloc[0]["close"] * 1.001:
                                short_term_trend = "BULLISH"
                            elif last_12.iloc[-1]["close"] < last_12.iloc[0]["close"] * 0.999:
                                short_term_trend = "BEARISH"
                        
                        # Long-term trend from daily close vs open
                        if latest["close"] > latest["open"] * 1.005:
                            long_term_trend = "BULLISH"
                        elif latest["close"] < latest["open"] * 0.995:
                            long_term_trend = "BEARISH"
                        
                        change_pct = ((latest["close"] - latest["open"]) / latest["open"]) * 100
                        
                        summary["instruments"][instrument] = {
                            "open": latest["open"],
                            "high": latest["high"],
                            "low": latest["low"],
                            "close": latest["close"],
                            "change_pct": change_pct,
                            "pdh": pdh_pdl.get("pdh") if pdh_pdl else None,
                            "pdl": pdh_pdl.get("pdl") if pdh_pdl else None,
                            "short_term_trend": short_term_trend,
                            "long_term_trend": long_term_trend,
                        }

                        # NEW: Option Chain Stats for EOD
                        try:
                            oc_data = self.option_fetcher.fetch_option_chain(instrument)
                            if oc_data:
                                pcr = self.option_analyzer.calculate_pcr(oc_data)
                                max_pain = self.option_analyzer.calculate_max_pain(oc_data)
                                spot = latest["close"]
                                oi_data = self.option_analyzer.analyze_oi_change(oc_data, spot)
                                
                                summary["instruments"][instrument]["option_chain"] = {
                                    "pcr": pcr,
                                    "max_pain": max_pain,
                                    "sentiment": oi_data.get("sentiment", "NEUTRAL"),
                                }
                        except Exception as e:
                            logger.error(f"⚠️ Failed to add option stats to summary for {instrument}: {e}")
                        
                except Exception as e:
                    logger.error(f"Error getting summary for {instrument}: {str(e)}")
            
            # Get AI forecast
            summary["ai_forecast"] = self._get_ai_market_forecast(summary)
            
            # Get performance stats
            summary["performance"] = self.trade_tracker.get_stats(days=1)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate daily summary: {str(e)}")
            return {}

    def _get_ai_market_forecast(self, summary: Dict) -> Dict:
        """Get AI-powered market forecast for next trading day."""
        try:
            # Build context for AI
            context_parts = []
            
            for inst, data in summary.get("instruments", {}).items():
                context_parts.append(
                    f"{inst}: Close {data['close']:.2f} ({data['change_pct']:+.2f}%), "
                    f"Trend: {data['short_term_trend']}/{data['long_term_trend']}"
                )
            
            stats = summary.get("statistics", {})
            context_parts.append(
                f"Today's activity: {stats.get('breakouts', 0)} breakouts, "
                f"{stats.get('breakdowns', 0)} breakdowns, {stats.get('retests', 0)} retests"
            )
            
            context = ". ".join(context_parts)
            
            forecast = self.ai_analyzer.forecast_market_outlook(context)
            return forecast
            
        except Exception as e:
            logger.warning(f"AI forecast failed: {str(e)}")
            return {"outlook": "NEUTRAL", "confidence": 50, "summary": "Forecast unavailable"}

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


    def check_scheduled_messages(self):
        """
        Check and send scheduled messages based on time windows.
        Uses persistence to ensuring messages are sent exactly once per day.
        """
        tz = pytz.timezone(TIME_ZONE)
        now = datetime.now(tz).time()
        
        # Get today's stats to check what has been sent
        daily_stats = self.persistence.get_daily_stats()
        
        # 1. Startup Message + PDH/PDL (09:15 onwards)
        # We allow a window until 09:30 to catch up if we missed 09:15
        if time(9, 15) <= now < time(9, 30):
            if not daily_stats.get("startup_msg_sent"):
                logger.info("⏰ Triggering Startup Message")
                
                pdh_pdl_stats = {}
                for instrument in INSTRUMENTS:
                    if INSTRUMENTS[instrument]["active"]:
                        stats = self.fetcher.get_previous_day_stats(instrument)
                        if stats:
                            pdh_pdl_stats[instrument] = stats
                
                if self.telegram_bot.send_startup_message(pdh_pdl_stats):
                    self.persistence.increment_stat("startup_msg_sent")
                else:
                    logger.warning("⚠️ Startup message failed to send")

        # 2. Market Context (09:30 onwards)
        elif time(9, 30) <= now < time(10, 0): # Window until 10:00 AM
            if not daily_stats.get("market_context_msg_sent"):
                logger.info("⏰ Triggering Market Context Update")
                
                context_data = {}
                pdh_pdl_stats = {}
                sr_levels = {}
                option_stats = {}
                
                # We need at least SOME data to send the message
                has_data = False

                for instrument in INSTRUMENTS:
                    if INSTRUMENTS[instrument]["active"]:
                        # Opening range stats
                        stats = self.fetcher.get_opening_range_stats(instrument)
                        if stats:
                            context_data[instrument] = stats
                            has_data = True
                        
                        # PDH/PDL
                        p_stats = self.fetcher.get_previous_day_stats(instrument)
                        if p_stats:
                            pdh_pdl_stats[instrument] = p_stats
                        
                        # NEW: S/R levels
                        try:
                            df_5m = self.fetcher.fetch_historical_data(instrument, period="5d", interval="5m")
                            if df_5m is not None and not df_5m.empty:
                                df_5m = self.fetcher.preprocess_ohlcv(df_5m)
                                analyzer = TechnicalAnalyzer(instrument)
                                sr = analyzer.calculate_support_resistance(df_5m)
                                sr_levels[instrument] = sr
                                has_data = True
                        except Exception as e:
                            logger.error(f"❌ S/R calculation failed for {instrument}: {e}")

                        # NEW: Option Chain Analysis
                        try:
                            oc_data = self.option_fetcher.fetch_option_chain(instrument)
                            if oc_data:
                                pcr = self.option_analyzer.calculate_pcr(oc_data)
                                max_pain = self.option_analyzer.calculate_max_pain(oc_data)
                                key_strikes = self.option_analyzer.get_key_strikes(oc_data)
                                
                                option_stats[instrument] = {
                                    "pcr": pcr,
                                    "max_pain": max_pain,
                                    "key_strikes": key_strikes
                                }
                                has_data = True
                        except Exception as e:
                            logger.error(f"❌ Option stats failed for {instrument}: {e}")
                
                if has_data:
                    if self.telegram_bot.send_market_context(context_data, pdh_pdl_stats, sr_levels, option_stats):
                        self.persistence.increment_stat("market_context_msg_sent")
                else:
                    logger.error("❌ No market data available for 9:30 update")
                    # Send partial/error notification if it's getting late (e.g. 09:40)
                    if now >= time(9, 40) and not daily_stats.get("market_context_error_sent"):
                        self.telegram_bot.send_error_notification(
                            "Failed to fetch market data for 9:30 update. Retrying..."
                        )
                        self.persistence.increment_stat("market_context_error_sent")

        # 3. End-of-Day Summary
        elif time(15, 31) <= now <= time(18, 0): # Wider window for EOD
             if not daily_stats.get("daily_summary_msg_sent"):
                logger.info("⏰ Triggering End-of-Day Summary")
                
                summary = self.generate_daily_summary()
                if summary:
                    if self.telegram_bot.send_daily_summary(summary):
                        self.persistence.increment_stat("daily_summary_msg_sent")

    def get_statistics(self) -> Dict:
        """Return simple statistics."""
        return {
            "signals_generated": len(self.signals_generated),
            "alerts_sent": self.alerts_sent,
            "ai_usage": self.ai_analyzer.get_usage_stats(),
            "bot_stats": self.telegram_bot.get_stats(),
        }


