"""
Signal Pipeline Module
Encapsulates the logic for filtering, scoring, and enriching trading signals.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pytz

from config.settings import (
    MAX_SAME_DIRECTION_ALERTS,
    MIN_SIGNAL_CONFIDENCE,
    TIME_ZONE,
)

logger = logging.getLogger(__name__)

class SignalPipeline:
    """
    Orchestrates the signal processing pipeline:
    1. Structural Checks
    2. Choppy Session Filter
    3. IV/Volatility Checks
    4. Correlation Checks
    5. Scoring & AI Enrichment
    """

    def __init__(self, groq_analyzer=None):
        self.groq_analyzer = groq_analyzer

    def process_signals(
        self,
        raw_signals: List[Dict],
        instrument: str,
        technical_context: Dict,
        option_metrics: Dict,
        recent_alerts: Dict,
        market_status: Dict
    ) -> List[Dict]:
        """
        Main entry point to process a batch of raw signals.
        """
        if not raw_signals:
            return []

        # Step 1: Choppy Session Filter & IV Check (Global Checks)
        # --------------------------------------------------------
        if market_status.get("is_choppy"):
            logger.info(f"⏭️  Suppressing signals for {instrument} (Choppy Session: {market_status.get('choppy_reason')})")
            return []

        iv = option_metrics.get("iv")
        if iv is not None and iv < 10:
             logger.info(f"⏭️  Suppressing signals for {instrument} (Low IV: {iv}%)")
             return []

        # Step 2: Correlation Check (Recent Alerts)
        # -----------------------------------------
        # Check if we have too many recent alerts in the same direction
        # This is a heuristic to prevent spamming "LONG" alerts if we just sent one.
        
        # Determine aggregate direction of new signals (if all are LONG or all are SHORT)
        has_long = any("BULLISH" in s["signal_type"] or "SUPPORT" in s["signal_type"] for s in raw_signals)
        has_short = any("BEARISH" in s["signal_type"] or "RESISTANCE" in s["signal_type"] for s in raw_signals)
        
        direction_check = "NEUTRAL"
        if has_long and not has_short: direction_check = "BULLISH"
        elif has_short and not has_long: direction_check = "BEARISH"
        
        if direction_check != "NEUTRAL":
            ist = pytz.timezone(TIME_ZONE)
            now = datetime.now(ist)
            recent_window = now - timedelta(minutes=15)
            
            recent_count = sum(
                1 for key, ts in recent_alerts.items()
                if ts > recent_window and 
                (getattr(key, "instrument", "") == instrument or isinstance(key, str) and instrument in key) and
                (
                    ("BULLISH" in str(key) or "SUPPORT" in str(key)) if direction_check == "BULLISH" else
                    ("BEARISH" in str(key) or "RESISTANCE" in str(key))
                )
            )
            
            if recent_count >= MAX_SAME_DIRECTION_ALERTS:
                # NEW: Check for fresh market structure
                has_new_structure = self._validate_fresh_structure(
                    raw_signals,
                    technical_context,
                    direction_check
                )
                
                if not has_new_structure:
                    logger.info(f"⏭️  Suppressing {direction_check} signals (Correlation Limit + No New Structure)")
                    return []
                else:
                    logger.info(f"✅ Allowing {direction_check} signal (Fresh Structure Detected)")


        # Step 3: Conflict Resolution
        # ---------------------------
        valid_signals = self.resolve_conflicts(raw_signals, option_metrics)
        
        processed_signals = []

        # Step 4: Individual Signal Scoring & AI
        # --------------------------------------
        for signal in valid_signals:
            # Calculate Technical Score
            score, reasons = self.calculate_score(signal, technical_context, option_metrics)
            signal["score"] = score
            signal["score_reasons"] = reasons
            
            # Filter by Tech Score
            # TODO: Move threshold to config (60)
            if score < 60:
                logger.debug(f"🛑 Signal Low Score: {score} | {signal['signal_type']}")
                continue
                
            # AI Enrichment (Advisory)
            if self.groq_analyzer:
                 try:
                    ai_context = {
                        "trend_direction": technical_context.get("higher_tf_context", {}).get("trend_direction", "FLAT"),
                        "pcr": option_metrics.get("pcr"),
                        "oi_sentiment": option_metrics.get("oi_change", {}).get("sentiment")
                    }
                    ai_analysis = self.groq_analyzer.analyze_signal(signal, ai_context, {})
                    signal["ai_analysis"] = ai_analysis
                    
                    # Log if AI disagrees strongly (but don't block by default)
                    if ai_analysis and "STRONG" in ai_analysis.get("verdict", "") and ai_analysis.get("confidence", 0) < 40:
                        logger.warning(f"⚠️ AI Disagrees with signal: {ai_analysis.get('verdict')}")
                        
                 except Exception as e:
                     logger.warning(f"⚠️ AI Analysis failed: {e}")
            
            processed_signals.append(signal)

        return processed_signals
    
    def _validate_fresh_structure(self, signals: List[Dict], context: Dict, direction: str) -> bool:
        """
        Check if signal represents fresh market structure.
        
        Returns True if:
        - New higher-high (for LONG) or lower-low (for SHORT)
        - VWAP reclaim
        - ORB high/low break
        - Volume surge (2x average)
        """
        try:
            df = context.get("df_5m")
            if df is None or len(df) < 2:
                return False
            
            current_price = df["close"].iloc[-1]
            vwap = context.get("vwap_5m", 0)
            
            if direction == "BULLISH":
                # Check for higher high
                recent_high = df["high"].iloc[-10:].max()
                if current_price > recent_high:
                    logger.debug(f"✅ New Higher High detected: {current_price:.2f}")
                    return True
                
                # Check for VWAP reclaim
                if vwap > 0:
                    prev_close = df["close"].iloc[-2]
                    if current_price > vwap and prev_close < vwap:
                        logger.debug(f"✅ VWAP Reclaim detected ({vwap:.2f})")
                        return True
            
            elif direction == "BEARISH":
                # Check for lower low
                recent_low = df["low"].iloc[-10:].min()
                if current_price < recent_low:
                    logger.debug(f"✅ New Lower Low detected: {current_price:.2f}")
                    return True
                
                # Check for VWAP breakdown
                if vwap > 0:
                    prev_close = df["close"].iloc[-2]
                    if current_price < vwap and prev_close > vwap:
                        logger.debug(f"✅ VWAP Breakdown detected ({vwap:.2f})")
                        return True
            
            # Check volume surge
            if "volume" in df.columns and df["volume"].sum() > 0:
                avg_volume = df["volume"].iloc[-20:-1].mean()
                current_volume = df["volume"].iloc[-1]
                if current_volume > avg_volume * 2.0:
                    logger.debug(f"✅ Volume Surge detected: {current_volume:.0f} vs avg {avg_volume:.0f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Fresh structure validation failed: {e}")
            return False

    def resolve_conflicts(self, signals: List[Dict], option_metrics: Dict) -> List[Dict]:
        """
        Resolve conflicting signals (LONG vs SHORT) using Option Data & Confidence.
        """
        if not signals:
            return []
            
        long_signals = [s for s in signals if any(x in s["signal_type"] for x in ["BULLISH", "SUPPORT"])]
        short_signals = [s for s in signals if any(x in s["signal_type"] for x in ["BEARISH", "RESISTANCE"])]
        
        if not long_signals or not short_signals:
            return signals # No conflict
            
        # Conflict Detected
        logger.info(f"⚔️ Conflict Detected: {len(long_signals)} LONG vs {len(short_signals)} SHORT")
        
        # 1. Option Chain Bias
        pcr = option_metrics.get("pcr")
        oi_sentiment = option_metrics.get("oi_change", {}).get("sentiment", "NEUTRAL")
        
        if oi_sentiment == "BULLISH" or (pcr and pcr > 1.2):
            logger.info("✅ Resolving to LONG (Option Bias)")
            return long_signals
            
        elif oi_sentiment == "BEARISH" or (pcr and pcr < 0.8):
            logger.info("✅ Resolving to SHORT (Option Bias)")
            return short_signals
            
        # 2. Fallback: Highest Technical Confidence
        all_sigs = long_signals + short_signals
        best_signal = max(all_sigs, key=lambda x: x.get('confidence', 0))
        logger.info(f"✅ Resolving to Highest Confidence: {best_signal['signal_type']}")
        return [best_signal]

    def calculate_score(self, sig_data: Dict, analysis_context: Dict, option_metrics: Dict) -> Tuple[int, List[str]]:
        """
        Pure function to calculate quality score (0-100).
        """
        score = 50  # Base Score
        reasons = []
        
        # 1. Price Action / Confidence
        if sig_data.get("confidence", 0) >= 80:
            score += 15
            reasons.append("Strong Pattern (+15)")
        elif sig_data.get("confidence", 0) >= 65:
            score += 10
            reasons.append("Good Pattern (+10)")

        if sig_data.get("volume_confirmed"):
            score += 10
            reasons.append("Volume High (+10)")
            
        # 2. Option Chain Sentiment
        pcr = option_metrics.get("pcr")
        oi_sentiment = option_metrics.get("oi_change", {}).get("sentiment", "NEUTRAL")
        signal_type = sig_data["signal_type"]
        is_bullish = "BULLISH" in signal_type or "SUPPORT" in signal_type
        
        if pcr:
            if is_bullish:
                if pcr > 1.0: 
                    score += 10
                    reasons.append(f"PCR Bullish {pcr} (+10)")
                elif pcr < 0.6: 
                    score -= 10
                    reasons.append(f"PCR Bearish {pcr} (-10)")
            else: # Bearish
                if pcr < 0.7: 
                    score += 10
                    reasons.append(f"PCR Bearish {pcr} (+10)")
                elif pcr > 1.2: 
                    score -= 10
                    reasons.append(f"PCR Bullish {pcr} (-10)")

        # OI Sentiment Logic
        if is_bullish and oi_sentiment == "BULLISH":
            score += 15
            reasons.append("OI Sentiment Bullish (+15)")
        elif not is_bullish and oi_sentiment == "BEARISH":
            score += 15
            reasons.append("OI Sentiment Bearish (+15)")
        elif (is_bullish and oi_sentiment == "BEARISH") or (not is_bullish and oi_sentiment == "BULLISH"):
            score -= 15
            reasons.append(f"OI Sentiment Conflict {oi_sentiment} (-15)")

        # 3. Multi-Timeframe Trend
        mtf_trend = analysis_context.get("higher_tf_context", {}).get("trend_direction", "FLAT")
        if mtf_trend != "FLAT":
            if is_bullish and mtf_trend == "UP":
                score += 15
                reasons.append("Trend Aligned (15m UP) (+15)")
            elif not is_bullish and mtf_trend == "DOWN":
                score += 15
                reasons.append("Trend Aligned (15m DOWN) (+15)")
            elif (is_bullish and mtf_trend == "DOWN") or (not is_bullish and mtf_trend == "UP"):
                score -= 10
                reasons.append(f"Counter Trend (15m {mtf_trend}) (-10)")
                
                # Reversal Boost
                is_reversal = any(x in sig_data["signal_type"] for x in ["BOUNCE", "RETEST", "PIN_BAR"])
                if is_reversal and (sig_data.get("volume_confirmed") or sig_data.get("momentum_confirmed")):
                     score += 10
                     reasons.append("Reversal Setup Bonus (+10)")

        return max(0, min(100, score)), reasons
