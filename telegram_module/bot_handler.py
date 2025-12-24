"""
Telegram Bot Handler
Sends formatted alerts, signals, and notifications to Telegram.
"""

import requests
import json
import logging
from typing import Dict, Optional
from datetime import datetime

from config.settings import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_CHANNEL_ID,
    TELEGRAM_MAX_MESSAGE_LENGTH,
    INCLUDE_CHARTS_IN_ALERT,
    INCLUDE_AI_SUMMARY_IN_ALERT,
    ALERT_TYPES,
    DRY_RUN,
    DEBUG_MODE,
)

logger = logging.getLogger(__name__)


class TelegramBot:
    """Send alerts and messages to Telegram."""

    def __init__(self):
        self.token = TELEGRAM_BOT_TOKEN
        self.chat_id = TELEGRAM_CHAT_ID
        self.channel_id = TELEGRAM_CHANNEL_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.message_count = 0

        logger.info(f"🤖 TelegramBot initialized | Chat ID: {self.chat_id} | Channel: {self.channel_id}")
        self._validate_credentials()

    def _validate_credentials(self):
        if not self.token or self.token == "YOUR_BOT_TOKEN_HERE":
            logger.error("❌ TELEGRAM_BOT_TOKEN not configured")
        if not self.chat_id or self.chat_id == "YOUR_CHAT_ID_HERE":
            logger.error("❌ TELEGRAM_CHAT_ID not configured")

    # =====================================================================
    # CORE SEND
    # =====================================================================

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a plain message to Telegram chat.
        """
        if DRY_RUN:
            logger.warning(f"🚫 DRY RUN: Not sending Telegram message: {text[:80]}...")
            return True

        if len(text) > TELEGRAM_MAX_MESSAGE_LENGTH:
            logger.warning(
                f"⚠️  Message too long ({len(text)} chars), truncating"
            )
            text = text[: TELEGRAM_MAX_MESSAGE_LENGTH - 3] + "..."

        import time

        url = f"{self.base_url}/sendMessage"
        
        # List of targets with labels for better logging
        targets = [(self.chat_id, "chat")]
        if self.channel_id:
            targets.append((self.channel_id, "channel"))
        
        success_count = 0
        failed_targets = []
        
        for target_id, target_type in targets:
            retries = 3
            backoff = 2
            last_error = None
            
            while retries > 0:
                try:
                    payload = {
                        "chat_id": target_id,
                        "text": text,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True,
                    }

                    response = requests.post(url, json=payload, timeout=10)
                    
                    # Handle Rate Limiting (429) explicitly
                    if response.status_code == 429:
                        retry_after = int(response.json().get("parameters", {}).get("retry_after", backoff))
                        logger.warning(f"⏳ Rate limited by Telegram ({target_type}). Retrying after {retry_after}s...")
                        time.sleep(retry_after)
                        retries -= 1
                        continue
                        
                    response.raise_for_status()

                    data = response.json()
                    if data.get("ok"):
                        msg_id = data["result"]["message_id"]
                        success_count += 1
                        logger.info(f"✅ Telegram message sent to {target_type} ({target_id}) | ID: {msg_id}")
                        break # Success, move to next target
                    else:
                        error_desc = data.get('description', 'Unknown')
                        last_error = f"{target_type}: {error_desc}"
                        logger.error(f"❌ Failed to send to {target_type} ({target_id}): {error_desc}")
                        failed_targets.append(last_error)
                        break # Logic error, don't retry

                except requests.exceptions.RequestException as e:
                    last_error = f"{target_type}: {str(e)}"
                    logger.error(f"❌ Telegram send to {target_type} failed (Attempt {4-retries}/3): {str(e)}")
                    retries -= 1
                    if retries > 0:
                        time.sleep(backoff)
                        backoff *= 2  # Exponential backoff for network errors
                    else:
                        # All retries exhausted
                        failed_targets.append(last_error)
        
        if success_count > 0:
            self.message_count += 1
            if failed_targets:
                logger.warning(f"⚠️ Partial success: {success_count}/{len(targets)} targets succeeded. Failed: {', '.join(failed_targets)}")
            return True
        else:
            # All targets failed
            logger.error(f"❌ All Telegram targets failed. Errors: {', '.join(failed_targets) if failed_targets else 'Unknown'}")
            return False

    # =====================================================================
    # SIGNAL ALERTS
    # =====================================================================

    def _format_targets(self, signal: Dict) -> str:
        """Format T1, T2, T3 targets."""
        tp1 = float(signal.get("take_profit", 0.0))
        tp2 = float(signal.get("take_profit_2", 0.0))
        tp3 = float(signal.get("take_profit_3", 0.0))
        
        # Base T1
        txt = f"🎯 Target 1: {tp1:.2f} (Safe)"
        
        # If strong trend, show T2/T3
        if tp2 > 0 and abs(tp2 - tp1) > 1.0:
             txt += f"\n🎯 Target 2: {tp2:.2f}"
        if tp3 > 0 and abs(tp3 - tp2) > 1.0:
             txt += f"\n🎯 Target 3: {tp3:.2f}"
             
        return txt

    def _format_ai_analysis(self, signal: Dict) -> str:
        """Helper to format AI analysis section."""
        if not INCLUDE_AI_SUMMARY_IN_ALERT:
            return ""
            
        ai_data = signal.get("ai_analysis")
        if not ai_data:
            return ""

        verdict = ai_data.get("verdict", "N/A")
        reasoning = ai_data.get("reasoning", "No details")
        ai_conf = ai_data.get("confidence", 0)
        ai_provider = ai_data.get("ai_provider", "GROQ")  # Default to GROQ if not specified
        
        # Check for legacy schema fallback
        if "recommendation" in ai_data:
            verdict = ai_data.get("recommendation")
            reasoning = ai_data.get("summary")
        
        # Provider emoji
        provider_emoji = "🔮" if ai_provider == "VERTEX" else "🧠"

        return (
            f"🤖 <b>AI Analyst Review</b> ({provider_emoji} {ai_provider})\n"
            f"• Verdict: {verdict} ({ai_conf}%)\n"
            f"• Logic: <i>{reasoning}</i>\n"
        )

    def send_breakout_alert(self, signal: Dict) -> bool:
        """Send formatted breakout/breakdown alert."""
        try:
            instrument = signal.get("instrument", "N/A")
            signal_type = signal.get("signal_type", "BREAKOUT")
            entry = float(signal.get("entry_price", 0.0))
            sl = float(signal.get("stop_loss", 0.0))
            tp = float(signal.get("take_profit", 0.0))
            rr = float(signal.get("risk_reward_ratio", 0.0))
            conf = float(signal.get("confidence", 0.0))

            emoji = "🚀" if "BULLISH" in signal_type else "📉"

            header = ALERT_TYPES.get(
                "BREAKOUT", "BREAKOUT"
            ) if "BREAKOUT" in signal_type else ALERT_TYPES.get(
                "BREAKDOWN", "BREAKDOWN"
            )

            message = (
                f"{emoji} {header}\n\n"
                f"📊 {instrument}\n"
                f"💰 Entry: {entry:.2f}\n"
                f"🛑 SL: {sl:.2f}\n"
                f"{self._format_targets(signal)}\n"
                f"📈 RR: {rr:.2f}:1\n"
                f"⚡ Confidence: {conf:.1f}%\n"
                f"🏆 Score: {signal.get('score', 'N/A')}/100\n\n"
                f"{signal.get('description', '')}\n\n"
            )
            
            # Add Score Reasons
            reasons = signal.get("score_reasons", [])
            if reasons:
                message += f"📝 Factors: {', '.join(reasons)}\n\n"
            
            # Add IST timestamp
            import pytz
            ist = pytz.timezone("Asia/Kolkata")
            now_ist = datetime.now(ist)
            message += f"⏰ {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}\n\n"

            # Add AI Analysis
            message += self._format_ai_analysis(signal)

            return self.send_message(message)

        except Exception as e:
            logger.error(f"❌ Failed to format breakout alert: {str(e)}", exc_info=True)
            logger.error(f"   Signal data causing error: {signal}")
            return False

    def send_false_breakout_alert(self, signal: Dict) -> bool:
        """Send false breakout warning."""
        try:
            instrument = signal.get("instrument", "N/A")
            level = float(signal.get("price_level", 0.0))
            fb = signal.get("false_breakout_details", {})
            retrace = float(fb.get("retracement_pct", 0.0))

            message = (
                f"⚠️ {ALERT_TYPES.get('FALSE_BREAKOUT', 'FALSE BREAKOUT')}\n\n"
                f"📊 {instrument}\n"
                f"📍 Level: {level:.2f}\n"
                f"↩ Retracement: {retrace:.2f}%\n\n"
                f"Possible trap / failed breakout at key level.\n\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            return self.send_message(message)

        except Exception as e:
            logger.error(
                f"❌ Failed to format false breakout alert: {str(e)}"
            )
            return False

    def send_retest_alert(self, signal: Dict) -> bool:
        """Send retest setup alert."""
        try:
            import pytz
            
            instrument = signal.get("instrument", "N/A")
            signal_type = signal.get("signal_type", "RETEST")
            level = float(signal.get("price_level", 0.0))
            entry = float(signal.get("entry_price", 0.0))
            sl = float(signal.get("stop_loss", 0.0))
            tp = float(signal.get("take_profit", 0.0))
            conf = float(signal.get("confidence", 0.0))
            desc = signal.get("description", "")
            
            # Determine direction
            direction = "📈 LONG" if entry > sl else "📉 SHORT"
            emoji = "🎯" if "SUPPORT" in signal_type.upper() else "🔄"
            
            # Calculate R:R
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr = reward / risk if risk > 0 else 0
            
            # Get IST time
            ist = pytz.timezone("Asia/Kolkata")
            now_ist = datetime.now(ist)

            # Determine level type from description
            level_type = "Resistance" if "resistance" in desc.lower() or "PDH" in desc else "Support" if "support" in desc.lower() or "PDL" in desc else "Level"
            
            message = (
                f"{emoji} {ALERT_TYPES.get('RETEST', 'RETEST')} {direction}\n\n"
                f"📊 <b>{instrument}</b>\n"
                f"📍 Near {level_type}: {level:.2f}\n\n"
                f"<b>💰 Entry:</b> {entry:.2f}\n"
                f"<b>🛑 Stop Loss:</b> {sl:.2f}\n"
                f"<b>{self._format_targets(signal)}</b>\n"
                f"<b>📈 Risk:Reward:</b> 1:{rr:.1f}\n"
                f"<b>⚡ Confidence:</b> {conf:.0f}%\n"
                f"<b>🏆 Score:</b> {signal.get('score', 'N/A')}/100\n\n"
                f"💡 {desc}\n\n"
            )
            
            # Add Score Reasons
            reasons = signal.get("score_reasons", [])
            if reasons:
                message += f"📝 Factors: {', '.join(reasons)}\n\n"
            
            message += f"⏰ {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}\n\n"
            
            # Add AI Analysis
            message += self._format_ai_analysis(signal)

            return self.send_message(message)

        except Exception as e:
            logger.error(f"❌ Failed to format retest alert: {str(e)}", exc_info=True)
            logger.error(f"   Signal data causing error: {signal}")
            return False

    def send_inside_bar_alert(self, signal: Dict) -> bool:
        """Send inside bar setup alert."""
        try:
            import pytz
            
            instrument = signal.get("instrument", "N/A")
            entry = float(signal.get("entry_price", 0.0))
            sl = float(signal.get("stop_loss", 0.0))
            tp = float(signal.get("take_profit", 0.0))
            conf = float(signal.get("confidence", 0.0))
            desc = signal.get("description", "")
            
            # Calculate R:R
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            rr = reward / risk if risk > 0 else 0
            
            # Get IST time
            ist = pytz.timezone("Asia/Kolkata")
            now_ist = datetime.now(ist)

            message = (
                f"📊 {ALERT_TYPES.get('INSIDE_BAR', 'INSIDE BAR SETUP')}\n\n"
                f"📊 <b>{instrument}</b>\n\n"
                f"<b>💰 Entry:</b> {entry:.2f}\n"
                f"<b>🛑 Stop Loss:</b> {sl:.2f}\n"
                f"<b>{self._format_targets(signal)}</b>\n"
                f"<b>📈 Risk:Reward:</b> 1:{rr:.1f}\n"
                f"<b>⚡ Confidence:</b> {conf:.0f}%\n"
                f"<b>🏆 Score:</b> {signal.get('score', 'N/A')}/100\n\n"
                f"💡 {desc}\n\n"
            )

            # Add Score Reasons
            reasons = signal.get("score_reasons", [])
            if reasons:
                message += f"📝 Factors: {', '.join(reasons)}\n\n"
                
            message += f"⏰ {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}\n\n"
            
            # Add AI Analysis
            message += self._format_ai_analysis(signal)

            return self.send_message(message)

        except Exception as e:
            logger.error(
                f"❌ Failed to format inside bar alert: {str(e)}", exc_info=True
            )
            logger.error(f"   Signal data causing error: {signal}")
            return False

    # =====================================================================
    # OTHER NOTIFICATIONS
    # =====================================================================

    def send_daily_summary(self, summary_data: Dict) -> bool:
        """Send comprehensive end-of-day market summary."""
        try:
            message = "<b>📊 END-OF-DAY MARKET SUMMARY</b>\n"
            message += f"📅 {datetime.now().strftime('%B %d, %Y')}\n\n"
            
            # Price action for each instrument
            instruments_data = summary_data.get("instruments", {})
            for instrument, data in instruments_data.items():
                change_pct = data.get("change_pct", 0)
                emoji = "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➖"
                
                message += f"<b>{emoji} {instrument}</b>\n"
                message += f"Open: {data.get('open', 0):.2f} | High: {data.get('high', 0):.2f}\n"
                message += f"Low: {data.get('low', 0):.2f} | <b>Close: {data.get('close', 0):.2f}</b>\n"
                message += f"Change: <b>{change_pct:+.2f}%</b>\n"
                
                # Key levels
                if data.get("pdh") and data.get("pdl"):
                    message += f"PDH: {data['pdh']:.2f} | PDL: {data['pdl']:.2f}\n"
                
                # Trend
                st_trend = data.get("short_term_trend", "NEUTRAL")
                lt_trend = data.get("long_term_trend", "NEUTRAL")
                message += f"Trend: {st_trend} (ST) / {lt_trend} (LT)\n"
                
                # Option Chain Summary
                if "option_chain" in data:
                    oc = data["option_chain"]
                    message += f"Option Chain: PCR {oc.get('pcr', 'N/A')} | MP {oc.get('max_pain', 'N/A')} | {oc.get('sentiment', 'NEUTRAL')}\n"
                
                message += "\n"
            
            # Events summary block removed as per user request (redundant with Performance)

            
            # Performance stats
            perf = summary_data.get("performance", {})
            if perf and perf.get("total_alerts", 0) > 0:
                message += "<b>📊 Performance (Today)</b>\n"
                message += f"Total Alerts: {perf.get('total_alerts', 0)}\n"
                
                # Only show win rate if we have closed trades
                wins = perf.get("wins", 0)
                losses = perf.get("losses", 0)
                if wins + losses > 0:
                    message += f"Win Rate: {perf.get('win_rate', 0):.1f}% ({wins}W-{losses}L)\n"
                
                by_type = perf.get("by_type", {})
                if by_type:
                    message += "<i>By Setup:</i>\n"
                    for stype, data in by_type.items():
                        readable_type = stype.replace("_", " ").title()
                        count = data.get("count", 0)
                        message += f"- {readable_type}: {count}\n"
                message += "\n"
            
            # AI Forecast
            forecast = summary_data.get("ai_forecast", {})
            if forecast:
                outlook = forecast.get("outlook", "NEUTRAL")
                outlook_emoji = "🟢" if outlook == "BULLISH" else "🔴" if outlook == "BEARISH" else "🟡"
                
                message += f"<b>{outlook_emoji} AI Forecast - {outlook}</b>\n"
                message += f"Confidence: {forecast.get('confidence', 50):.0f}%\n"
                
                # Parse summary if it's a JSON string (common with LLM output)
                ai_summary = forecast.get("summary", "No forecast available")
                if isinstance(ai_summary, str) and ai_summary.strip().startswith("{"):
                    try:
                        import json
                        parsed = json.loads(ai_summary)
                        ai_summary = parsed.get("summary", ai_summary)
                    except:
                        pass
                        
                message += f"{ai_summary}\n\n"
            
            # Statistics
            stats = summary_data.get("stats", {})
            message += "<b>📊 Session Stats</b>\n"
            message += f"📡 Data Fetches: {stats.get('data_fetches', 0)}\n"
            message += f"🔍 Analyses: {stats.get('analyses_run', 0)}\n"
            message += f"🔔 Alerts Sent: {stats.get('alerts_sent', 0)}\n"
            
            return self.send_message(message)

        except Exception as e:
            logger.error(
                f"❌ Failed to send daily summary: {str(e)}"
            )
            return False

    def send_error_notification(
        self, error_msg: str, context: str = ""
    ) -> bool:
        """Send error notification."""
        try:
            message = (
                "❌ ERROR NOTIFICATION\n\n"
                f"{context}\n\n"
                f"Error: {error_msg[:500]}\n\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            return self.send_message(message)

        except Exception as e:
            logger.error(
                f"❌ Failed to send error notification: {str(e)}"
            )
            return False

    def send_startup_message(self, pdh_pdl_stats: Optional[Dict] = None) -> bool:
        """Send startup confirmation message with optional PDH/PDL stats."""
        try:
            message = (
                "🚀 NIFTY AI TRADING AGENT STARTED\n\n"
                "✅ System online and monitoring markets\n"
                "📊 Instruments: NIFTY50, BANKNIFTY\n"
                "🔔 Breakout / retest / inside bar alerts will be sent\n"
                "⏰ Active: 09:15 - 15:30 IST\n"
            )

            if pdh_pdl_stats:
                message += "\n📋 <b>Previous Day Stats</b>\n"
                for instrument, stats in pdh_pdl_stats.items():
                    message += (
                        f"\n<b>{instrument}</b>\n"
                        f"High: {stats['pdh']:.2f}\n"
                        f"Low: {stats['pdl']:.2f}\n"
                        f"Close: {stats['pdc']:.2f}\n"
                    )

            return self.send_message(message)

        except Exception as e:
            logger.error(
                f"❌ Failed to send startup message: {str(e)}"
            )
            return False

    def send_market_context(self, context_data: Dict, pdh_pdl_stats: Optional[Dict] = None, sr_levels: Optional[Dict] = None, option_stats: Optional[Dict] = None) -> bool:
        """Send market context (Opening Range + S/R) update with optional PDH/PDL."""
        try:
            message = "🌅 <b>MARKET CONTEXT UPDATE</b>\n\n"
            
            all_instruments = set(context_data.keys())
            if pdh_pdl_stats:
                all_instruments.update(pdh_pdl_stats.keys())
            if sr_levels:
                all_instruments.update(sr_levels.keys())
            if option_stats:
                all_instruments.update(option_stats.keys())
            
            for instrument in sorted(list(all_instruments)):
                message += f"<b>{instrument}</b>\n"
                
                # PDH/PDL
                if pdh_pdl_stats and instrument in pdh_pdl_stats:
                    stats = pdh_pdl_stats[instrument]
                    message += (
                        f"PDH: {stats['pdh']:.2f} | PDL: {stats['pdl']:.2f}\n"
                    )

                # Opening Range
                if instrument in context_data:
                    stats = context_data[instrument]
                    if "orb_5m_high" in stats:
                        message += (
                            f"5m OR: {stats['orb_5m_low']:.2f} - {stats['orb_5m_high']:.2f}\n"
                        )
                    if "orb_15m_high" in stats:
                        message += (
                            f"15m OR: {stats['orb_15m_low']:.2f} - {stats['orb_15m_high']:.2f}\n"
                        )
                
                # NEW: Support/Resistance Levels
                if sr_levels and instrument in sr_levels:
                    sr = sr_levels[instrument]
                    
                    # Handle both dictionary (tests) and TechnicalLevels object (prod)
                    if isinstance(sr, dict):
                        s_levels = sr.get('support', []) or sr.get('support_levels', [])
                        r_levels = sr.get('resistance', []) or sr.get('resistance_levels', [])
                    else:
                        # Assume TechnicalLevels dataclass
                        s_levels = getattr(sr, 'support_levels', [])
                        r_levels = getattr(sr, 'resistance_levels', [])

                    # Show top 3 supports and resistances
                    # Support = price floor BELOW current price = LOWEST values
                    # Resistance = price ceiling ABOVE current price = HIGHEST values
                    supports = sorted(s_levels)[:3] if s_levels else []  # Take FIRST 3 (lowest)
                    resistances = sorted(r_levels)[-3:] if r_levels else []  # Take LAST 3 (highest)
                    
                    if supports:
                        message += f"📊 Supports: {', '.join([f'{s:.2f}' for s in supports])}\n"
                    if resistances:
                        message += f"📊 Resistances: {', '.join([f'{r:.2f}' for r in resistances])}\n"

                # NEW: Option Chain Stats
                if option_stats and instrument in option_stats:
                    oc = option_stats[instrument]
                    message += f"🎲 PCR: {oc.get('pcr', 'N/A')} | Max Pain: {oc.get('max_pain', 'N/A')}\n"
                    
                    ks = oc.get('key_strikes', {})
                    if ks:
                         message += f"🔑 Res: {ks.get('max_call_oi_strike', 'N/A')} | Sup: {ks.get('max_put_oi_strike', 'N/A')}\n"
                
                message += "\n"

            message += f"⏰ {datetime.now().strftime('%H:%M:%S')}"
            return self.send_message(message)

        except Exception as e:
            logger.error(f"❌ Failed to send market context: {str(e)}")
            return False

    # =====================================================================
    # MEDIA (optional stub)
    # =====================================================================

    def send_chart(self, chart_path: str, caption: str = "") -> bool:
        """Send chart image to Telegram (optional, can be extended)."""
        if not INCLUDE_CHARTS_IN_ALERT:
            logger.debug("⏭️  Chart sending disabled in settings")
            return True

        if DRY_RUN:
            logger.warning(
                f"🚫 DRY RUN: Not sending chart: {chart_path}"
            )
            return True

        try:
            url = f"{self.base_url}/sendPhoto"
            
            targets = [self.chat_id]
            if self.channel_id:
                targets.append(self.channel_id)
            
            success_count = 0

            with open(chart_path, "rb") as photo:
                file_content = photo.read()
                
            for target_id in targets:
                # Re-open file or use content for each request? 
                # requests files param expects an open file handle. 
                # Better to send bytes.
                
                try:
                    files = {"photo": file_content}
                    data = {
                        "chat_id": target_id,
                        "caption": caption[:1024],
                    }

                    logger.debug(f"📤 Sending chart to {target_id}: {chart_path}")
                    response = requests.post(
                        url, files=files, data=data, timeout=30
                    )
                    response.raise_for_status()

                    if response.json().get("ok"):
                        success_count += 1
                        logger.info(f"✅ Chart sent to {target_id}")
                    else:
                        logger.error(f"❌ Failed to send chart to {target_id}")
                except Exception as inner_e:
                    logger.error(f"❌ Error sending chart to {target_id}: {inner_e}")

            return success_count > 0

        except Exception as e:
            logger.error(f"❌ Chart sending failed: {str(e)}")
            return False

    # =====================================================================
    # CONNECTION TEST & STATS
    # =====================================================================

    def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        try:
            logger.info("🧪 Testing Telegram connection...")
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("ok"):
                bot_name = data["result"].get("first_name", "Bot")
                logger.info(
                    f"✅ Telegram connection successful | Bot: {bot_name}"
                )
                return True
            logger.error("❌ Telegram getMe returned not ok")
            return False
        except Exception as e:
            logger.error(f"❌ Telegram connection test failed: {str(e)}")
            return False

    def get_stats(self) -> Dict:
        """Get simple bot statistics."""
        return {
            "messages_sent": self.message_count,
            "chat_id": self.chat_id,
        }


_bot: Optional[TelegramBot] = None


def get_bot() -> TelegramBot:
    """Singleton getter for TelegramBot."""
    global _bot
    if _bot is None:
        _bot = TelegramBot()
    return _bot


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    bot = get_bot()
    bot.test_connection()
    bot.send_startup_message()


# Alias for backward compatibility / explicit naming
TelegramBotHandler = TelegramBot

def format_signal_message(signal: Dict) -> str:
    """Legacy/Helper formatter (optional)."""
    return f"{signal.get('signal_type', 'SIGNAL')} @ {signal.get('price_level', 0)}"
