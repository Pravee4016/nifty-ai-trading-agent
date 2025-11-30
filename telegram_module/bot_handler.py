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
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.message_count = 0

        logger.info(f"🤖 TelegramBot initialized | Chat ID: {self.chat_id}")
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

        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }

            logger.debug(
                f"📤 Sending Telegram message | Length: {len(text)} chars"
            )
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get("ok"):
                msg_id = data["result"]["message_id"]
                self.message_count += 1
                logger.info(f"✅ Telegram message sent | ID: {msg_id}")
                if DEBUG_MODE:
                    logger.debug(f"   Preview: {text[:150]}...")
                return True

            logger.error(
                f"❌ Telegram API error: {data.get('description', 'Unknown')}"
            )
            return False

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Telegram send failed: {str(e)}")
            return False

    # =====================================================================
    # SIGNAL ALERTS
    # =====================================================================

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
                f"🎯 TP: {tp:.2f}\n"
                f"📈 RR: {rr:.2f}:1\n"
                f"⚡ Confidence: {conf:.1f}%\n\n"
                f"{signal.get('description', '')}\n\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            if signal.get("ai_analysis") and INCLUDE_AI_SUMMARY_IN_ALERT:
                ai_data = signal["ai_analysis"]
                summary = ai_data.get("summary", "")
                reco = ai_data.get("recommendation", "HOLD")
                ai_conf = ai_data.get("confidence", 0)
                message += (
                    f"\n\n🤖 AI: {reco} ({ai_conf:.0f}%)\n"
                    f"{summary[:300]}"
                )

            return self.send_message(message)

        except Exception as e:
            logger.error(f"❌ Failed to format breakout alert: {str(e)}")
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
            instrument = signal.get("instrument", "N/A")
            level = float(signal.get("price_level", 0.0))
            desc = signal.get("description", "")

            message = (
                f"{ALERT_TYPES.get('RETEST', 'RETEST')}\n\n"
                f"📊 {instrument}\n"
                f"📍 Level: {level:.2f}\n\n"
                f"{desc}\n\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            return self.send_message(message)

        except Exception as e:
            logger.error(f"❌ Failed to format retest alert: {str(e)}")
            return False

    def send_inside_bar_alert(self, signal: Dict) -> bool:
        """Send inside bar setup alert."""
        try:
            instrument = signal.get("instrument", "N/A")
            desc = signal.get("description", "")

            message = (
                f"{ALERT_TYPES.get('INSIDE_BAR', 'INSIDE BAR')}\n\n"
                f"📊 {instrument}\n\n"
                f"{desc}\n\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            return self.send_message(message)

        except Exception as e:
            logger.error(
                f"❌ Failed to format inside bar alert: {str(e)}"
            )
            return False

    # =====================================================================
    # OTHER NOTIFICATIONS
    # =====================================================================

    def send_daily_summary(self, summary_data: Dict) -> bool:
        """Send daily market summary."""
        try:
            instrument = summary_data.get("instrument", "N/A")
            message = (
                "📊 DAILY MARKET SUMMARY\n\n"
                f"Instrument: {instrument}\n"
                f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
                f"Open: {summary_data.get('open')}\n"
                f"High: {summary_data.get('high')}\n"
                f"Low: {summary_data.get('low')}\n"
                f"Close: {summary_data.get('close')}\n\n"
                f"Signals Generated: {summary_data.get('signal_count', 0)}"
            )

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

    def send_startup_message(self) -> bool:
        """Send startup confirmation message."""
        try:
            message = (
                "🚀 NIFTY AI TRADING AGENT STARTED\n\n"
                "✅ System online and monitoring markets\n"
                "📊 Instruments: NIFTY50, BANKNIFTY\n"
                "🔔 Breakout / retest / inside bar alerts will be sent\n"
                "⏰ Active: 09:15 - 15:30 IST\n"
            )

            return self.send_message(message)

        except Exception as e:
            logger.error(
                f"❌ Failed to send startup message: {str(e)}"
            )
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
            with open(chart_path, "rb") as photo:
                files = {"photo": photo}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption[:1024],
                }

                logger.debug(f"📤 Sending chart: {chart_path}")
                response = requests.post(
                    url, files=files, data=data, timeout=30
                )
                response.raise_for_status()

                if response.json().get("ok"):
                    logger.info("✅ Chart sent to Telegram")
                    return True

                logger.error("❌ Failed to send chart")
                return False

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
