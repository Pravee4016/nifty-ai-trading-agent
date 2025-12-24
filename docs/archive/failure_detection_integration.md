# Signal Failure Detection - Implementation Guide

## What We Have

✅ **Created**: `signal_failure_detector.py` module
- Tracks last 5 signals per instrument in Firestore
- Checks if signals are failing (price moving against direction)
- Detects failures within ~0.15% movement or 50% to stop loss

✅ **Initialized**: Failure detector in agent

## Manual Integration Steps (5 minutes)

### Step 1: Add failure checking to run_analysis

In `/Users/praveent/nifty-ai-trading-agent/app/agent.py`, around line 148-151:

**FIND:**
```python
            logger.info(f"\\n🔍 Analyzing: {instrument}")
            logger.info("-" * 70)

            instrument_result = self._analyze_single_instrument(instrument)
```

**ADD THIS LINE AFTER line 149:**
```python
            logger.info("-" * 70)
            
            # Check for failing signals before new analysis
            self._check_failing_signals(instrument)  # ← ADD THIS LINE
            
            instrument_result = self._analyze_single_instrument(instrument)
```

---

### Step 2: Track signals after sending alerts

In `/Users/praveent/nifty-ai-trading-agent/app/agent.py`, find the `_send_alert` method around line 640:

**FIND (around line 640):**
```python
            if success:
                logger.info("   ✅ Telegram alert sent")
                self.persistence.increment_stat("alerts_sent")
                
                # Record trade for performance tracking
                trade_id = self.trade_tracker.record_alert(signal)
```

**ADD AFTER `trade_id =` line:**
```python
                trade_id = self.trade_tracker.record_alert(signal)
                if trade_id:
                    logger.info(f"   📝 Trade tracked: {trade_id}")
                
                # NEW: Track signal for failure detection
                self._track_signal_for_failure(signal, instrument)  # ← ADD THIS LINE
```

---

### Step 3: Add helper methods

At the END of the `NiftyTradingAgent` class (before the last closing line), around line 999, ADD these three methods:

```python
    def _check_failing_signals(self, instrument: str) -> None:
        """Check if any recent signals are failing and send alerts."""
        try:
            # Get current price
            market_data = self.fetcher.fetch_realtime_data(instrument)
            if not market_data or 'price' not in market_data:
                logger.debug(f"⏭️ Skipping failure check for {instrument} - no price data")
                return
            
            current_price = market_data['price']
            
            # Check signal health
            failing_signals = self.failure_detector.check_signal_health(instrument, current_price)
            
            if failing_signals:
                logger.info(f"⚠️ Found {len(failing_signals)} failing signals for {instrument}")
                
                for failure in failing_signals:
                    # Send failure alert
                    self._send_failure_alert(failure, instrument)
                    
                    # Mark as alerted
                    self.failure_detector.mark_signal_alerted(failure['doc_id'], instrument)
            
        except Exception as e:
            logger.error(f"Failed to check failing signals: {e}")
    
    def _send_failure_alert(self, failure: Dict, instrument: str) -> None:
        """Send Telegram alert for a failing signal."""
        try:
            signal = failure['signal']
            current_price = failure['current_price']
            price_change = failure['price_change_pct']
            minutes_ago = failure['minutes_since_entry']
            
            # Format alert message
            direction_emoji = "📈" if signal['direction'] == 'LONG' else "📉"
            warning_emoji = "⚠️"
            
            message = f"{warning_emoji} **SIGNAL FAILING** {direction_emoji}\\n\\n"
            message += f"📊 **{instrument}**\\n"
            message += f"Signal: {signal['signal_type']}\\n"
            message += f"Direction: {signal['direction']}\\n\\n"
            
            message += f"🎯 Entry: {signal['entry_price']:.2f}\\n"
            message += f"💰 Current: {current_price:.2f}\\n"
            message += f"🛑 Stop Loss: {signal['stop_loss']:.2f}\\n\\n"
            
            message += f"📉 Change: {price_change:+.2f}%\\n"
            message += f"⏱️ Time: {minutes_ago:.0f} minutes ago\\n\\n"
            
            if price_change < -0.3:
                message += f"🚨 **URGENT**: Price moved significantly against signal!\\n"
            else:
                message += f"⚡ **Consider exit** if in position\\n"
            
            # Send via Telegram
            if self.telegram_bot.send_message(message):
                logger.info(f"   ✅ Failure alert sent for {signal['signal_type']}")
            
        except Exception as e:
            logger.error(f"Failed to send failure alert: {e}")
    
    def _track_signal_for_failure(self, signal: Dict, instrument: str) -> None:
        """Track a signal after sending alert for failure detection."""
        try:
            self.failure_detector.track_signal(signal, instrument)
        except Exception as e:
            logger.error(f"Failed to track signal: {e}")
```

---

## Testing

After making these changes:

```bash
cd /Users/praveent/nifty-ai-trading-agent
python main.py --once
```

**Look for:**
1. "✅ Signal Failure Detector initialized"
2. "📍 Tracking LONG/SHORT signal @ price"
3. "⚠️ Found X failing signals" (if any tracked signals are failing)

---

## Example Alert

When a signal fails, you'll get:

```
⚠️ **SIGNAL FAILING** 📉

📊 **NIFTY**
Signal: RESISTANCE_BOUNCE
Direction: SHORT

🎯 Entry: 26123.30
💰 Current: 26180.00
🛑 Stop Loss: 26199.53

📉 Change: -0.22%
⏱️ Time: 2 minutes ago

⚡ **Consider exit** if in position
```

---

## Quick Apply Script

Or run this to apply all changes automatically:

```bash
# Coming in next message - let me know if you want me to create a patch script
```

---

**Estimated time to apply manually**: 5-10 minutes  
**Benefit**: Catch failing signals within 2 minutes! 🎯
