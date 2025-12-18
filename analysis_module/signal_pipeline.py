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
    MIN_SCORE_THRESHOLD,
    TIME_ZONE,
    USE_ML_FILTERING,
    ML_MODEL_BUCKET,
    ML_MODEL_NAME,
    ML_CONFIDENCE_THRESHOLD,
    ML_FALLBACK_TO_RULES,
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
        
        # Initialize ML predictor if enabled
        self.ml_predictor = None
        if USE_ML_FILTERING:
            try:
                from ml_module.predictor import SignalQualityPredictor
                self.ml_predictor = SignalQualityPredictor(
                    bucket_name=ML_MODEL_BUCKET,
                    model_name=ML_MODEL_NAME
                )
                logger.info("✅ ML predictor initialized")
            except Exception as e:
                logger.warning(f"⚠️ ML predictor failed to initialize: {e}")
                if not ML_FALLBACK_TO_RULES:
                    logger.error("ML_FALLBACK_TO_RULES=False, will reject all signals")
                else:
                    logger.info("Falling back to rule-based scoring")
        
        # Initialize Market State Engine
        from analysis_module.market_state_engine import MarketStateEngine
        self.state_engine = MarketStateEngine()
        logger.info("✅ Market State Engine initialized")

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

        # Step 1: Market State Evaluation (NEW)
        # --------------------------------------------------------
        from analysis_module.market_state_engine import MarketState
        
        # Get dataframe from technical_context (should be passed)
        df = technical_context.get("df")  # OHLCV dataframe
        vwap_series = technical_context.get("vwap_series")  # Optional VWAP values
        
        if df is not None and len(df) >= 10:
            market_state_info = self.state_engine.evaluate_state(df, vwap_series)
            state = market_state_info["state"]
            confidence = market_state_info["confidence"]
            reasons = market_state_info["reasons"]
            
            logger.info(
                f"📊 Market State: {state.value} | "
                f"Confidence: {confidence:.0%} | "
                f"Reasons: {', '.join(reasons)}"
            )
        else:
            # Fallback to old choppy check if df not available
            if market_status.get("is_choppy"):
                logger.info(f"⏭️  Suppressing signals for {instrument} (Choppy Session: {market_status.get('choppy_reason')})")
                return []
            # Default to TRANSITION if no data
            state = MarketState.TRANSITION
            confidence = 0.5
            reasons = ["Using fallback state (no df)"]
            logger.warning("⚠️ No df in technical_context, using fallback TRANSITION state")
        
        # CHOPPY state: Block all signals
        if state == MarketState.CHOPPY:
            logger.info(f"🛑 CHOPPY State | Blocking all {len(raw_signals)} signals | "
                       f"Reasons: {', '.join(reasons)}")
            return []
        
        # IV Check (still relevant)
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
        
        # Step 3.5: State-Based Strategy Gating (NEW)
        # -------------------------------------------
        gated_signals = self._gate_signals_by_state(valid_signals, state)
        
        if len(gated_signals) < len(valid_signals):
            blocked = len(valid_signals) - len(gated_signals)
            logger.info(f"⏭️ {state.value} State | Blocked {blocked} signals (state-gated)")
        
        processed_signals = []

        # Step 4: Individual Signal Scoring & AI
        # --------------------------------------
        for signal in valid_signals:
            # ML-Based Scoring (if enabled and available)
            if self.ml_predictor and self.ml_predictor.enabled:
                try:
                    from ml_module.feature_extractor import extract_features
                    
                    # Extract features
                    features = extract_features(
                        signal,
                        technical_context,
                        option_metrics,
                        market_status
                    )
                    
                    # Get state-aware ML threshold
                    ml_threshold = self._get_ml_threshold_by_state(state)
                    
                    # Get ML prediction with state-aware threshold
                    should_accept, ml_prob = self.ml_predictor.predict_with_threshold(
                        features,
                        threshold=ml_threshold
                    )
                    
                    signal["ml_probability"] = ml_prob
                    signal["score"] = int(ml_prob * 100)  # Convert to 0-100 scale
                    signal["score_reasons"] = [
                        f"ML Win Probability: {ml_prob:.1%}",
                        f"State: {state.value} (threshold: {ml_threshold:.0%})"
                    ]
                    
                    if not should_accept:
                        logger.info(
                            f"🛑 ML Rejected | Prob: {ml_prob:.2%} < {ml_threshold:.0%} | "
                            f"{state.value} | {signal['signal_type']}"
                        )
                        continue
                    
                    logger.debug(f"✅ ML Accepted | Prob: {ml_prob:.2%} ({state.value}) | "
                                f"{signal['signal_type']}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ ML prediction failed: {e}")
                    if not ML_FALLBACK_TO_RULES:
                        logger.info("Skipping signal (no fallback enabled)")
                        continue
                    # Fall through to rule-based scoring below
            
            # Rule-Based Scoring (fallback or if ML disabled)
            if not self.ml_predictor or not self.ml_predictor.enabled or "score" not in signal:
                score, reasons = self.calculate_score(signal, technical_context, option_metrics)
                signal["score"] = score
                signal["score_reasons"] = reasons
                
                # Filter by Tech Score
                if score < MIN_SCORE_THRESHOLD:
                    logger.debug(f"🛑 Signal Low Score: {score} | {signal['signal_type']}")
                    continue
                
            # AI Enrichment (Advisory)
            if self.groq_analyzer:
                 try:
                    htf = technical_context.get("higher_tf_context", {})
                    oi_change = option_metrics.get("oi_change", {})
                    
                    ai_context = {
                        "trend_direction": htf.get("trend_direction", "FLAT"),
                        "trend_5m": htf.get("trend_5m", "NEUTRAL"),
                        "trend_15m": htf.get("trend_15m", "NEUTRAL"),
                        "trend_daily": htf.get("trend_daily", "NEUTRAL"),
                        "pcr": option_metrics.get("pcr"),
                        "iv": option_metrics.get("iv"),
                        "vix": htf.get("india_vix"),
                        "oi_sentiment": oi_change.get("sentiment"),
                        "above_vwap": htf.get("above_vwap"),
                        "above_ema20": htf.get("above_ema20"),
                        "pdh": htf.get("pdh"),
                        "pdl": htf.get("pdl"),
                        "market_state": state.value if hasattr(state, 'value') else str(state)
                    }
                    
                    # Pass the full technical_context for deeper access if needed
                    ai_analysis = self.groq_analyzer.analyze_signal(signal, ai_context, technical_context)
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

        # 4. Expert Indicators (RSI Divergence, Bollinger, EMA 50)
        htf = analysis_context.get("higher_tf_context", {})
        
        # RSI Divergence Bonus
        div_5m = htf.get("rsi_divergence_5m", "NONE")
        if div_5m != "NONE":
            if is_bullish and div_5m == "BULLISH":
                score += 15
                reasons.append("Bullish Divergence (5m) (+15)")
            elif not is_bullish and div_5m == "BEARISH":
                score += 15
                reasons.append("Bearish Divergence (5m) (+15)")
            else:
                score -= 10
                reasons.append(f"Divergence Conflict ({div_5m}) (-10)")

        # Bollinger Band Proximity
        current_price = sig_data.get("entry_price") or sig_data.get("price_level")
        bb_upper = htf.get("bb_upper_5m", 0.0)
        bb_lower = htf.get("bb_lower_5m", 0.0)
        
        if current_price and bb_upper and bb_lower:
            if is_bullish and current_price > bb_upper:
                score -= 10
                reasons.append("Overextended (Above BB Upper) (-10)")
            elif not is_bullish and current_price < bb_lower:
                score -= 10
                reasons.append("Overextended (Below BB Lower) (-10)")

        # EMA 50 Alignment (15m)
        ema_50_15m = htf.get("ema_50_15m", 0.0)
        if ema_50_15m > 0:
            if is_bullish and current_price > ema_50_15m:
                score += 10
                reasons.append("Above 15m EMA50 (+10)")
            elif not is_bullish and current_price < ema_50_15m:
                score += 10
                reasons.append("Below 15m EMA50 (+10)")

        # 5. Confluence Detection (EXPERT'S CORE METHODOLOGY)
        # Can be disabled via USE_EXPERT_ENHANCEMENTS=False
        from config.settings import USE_EXPERT_ENHANCEMENTS
        
        if USE_EXPERT_ENHANCEMENTS:
            from analysis_module.confluence_detector import detect_confluence
            
            # Get TechnicalLevels from analysis context
            levels = analysis_context.get("levels")
            if levels and current_price:
                try:
                    confluence_data = detect_confluence(
                        price=current_price,
                        levels=levels,
                        higher_tf_context=htf
                    )
                    
                    confluence_count = confluence_data.get('confluence_count', 0)
                    confluence_score_bonus = confluence_data.get('confluence_score', 0)
                    level_names = confluence_data.get('level_names', [])
                    
                    if confluence_count >= 3:
                        score += 25
                        reasons.append(f"HIGH Confluence ({confluence_count} levels: {', '.join(level_names[:3])}) (+25)")
                    elif confluence_count == 2:
                        score += 15
                        reasons.append(f"Confluence ({', '.join(level_names)}) (+15)")
                    elif confluence_count == 1:
                        score += 5
                        reasons.append(f"Near {level_names[0]} (+5)")
                        
                except Exception as e:
                    logger.warning(f"Confluence detection failed: {e}")

        # 6. Precision Entry Bonus (±3 Point Rule)
        # 7. Rejection Pattern at Confluence BONUS (Expert's Edge)
        # Both can be disabled via USE_EXPERT_ENHANCEMENTS=False
        if USE_EXPERT_ENHANCEMENTS:
            # Precision Entry Bonus
            if levels and current_price:
                try:
                    # Find nearest key level
                    key_levels = [
                        ('PDH', levels.pdh),
                        ('PDL', levels.pdl),
                        ('Fib_R1', levels.r1_fib),
                        ('Fib_S1', levels.s1_fib),
                        ('Fib_R2', levels.r2_fib),
                        ('Fib_S2', levels.s2_fib),
                    ]
                    
                    nearest_distance = float('inf')
                    nearest_level_name = None
                    
                    for name, level in key_levels:
                        if level > 0:
                            distance = abs(current_price - level)
                            if distance < nearest_distance:
                                nearest_distance = distance
                                nearest_level_name = name
                    
                    # Award bonus if within ±3 points
                    if nearest_distance <= 3.0:
                        score += 10
                        reasons.append(f"Precise Entry (±{nearest_distance:.1f}pts from {nearest_level_name}) (+10)")
                        
                except Exception as e:
                    logger.warning(f"Precision entry check failed: {e}")

            # Rejection Pattern at Confluence BONUS
            is_rejection = any(x in signal_type for x in ["PIN_BAR", "BOUNCE", "RETEST"])
            
            if is_rejection and levels and current_price:
                try:
                    # Check confluence for rejection patterns
                    from analysis_module.confluence_detector import detect_confluence
                    confluence_data = detect_confluence(
                        price=current_price,
                        levels=levels,
                        higher_tf_context=htf
                    )
                    
                    rejection_confluence = confluence_data.get('confluence_count', 0)
                    level_names = confluence_data.get('level_names', [])
                    
                    if rejection_confluence >= 2:
                        score += 20
                        reasons.append(f"🎯 Rejection at Confluence ({', '.join(level_names[:2])}) (+20)")
                    elif rejection_confluence == 1:
                        score += 10
                        reasons.append(f"Rejection at {level_names[0]} (+10)")
                        
                except Exception as e:
                    logger.warning(f"Rejection confluence bonus failed: {e}")

        return max(0, min(100, score)), reasons
    
    def _get_ml_threshold_by_state(self, state) -> float:
        """Get ML confidence threshold based on market state."""
        from analysis_module.market_state_engine import MarketState
        
        STATE_ML_THRESHOLDS = {
            MarketState.CHOPPY: None,
            MarketState.TRANSITION: 0.80,  # High bar
            MarketState.EXPANSIVE: 0.65    # Normal
        }
        
        return STATE_ML_THRESHOLDS.get(state, ML_CONFIDENCE_THRESHOLD)
    
    def _gate_signals_by_state(self, signals: List[Dict], state) -> List[Dict]:
        """Filter signals based on market state and strategy type."""
        from analysis_module.market_state_engine import MarketState
        
        if state == MarketState.CHOPPY:
            return []
        
        if state == MarketState.EXPANSIVE:
            return signals
        
        # TRANSITION: Selective strategies only
        gated = []
        for signal in signals:
            signal_type = signal.get("signal_type", "")
            
            if "BREAKOUT" in signal_type or "BREAKDOWN" in signal_type:
                gated.append(signal)
            elif "RETEST" in signal_type or "BOUNCE" in signal_type:
                logger.debug(f"🚧 TRANSITION | Blocking {signal_type}")
            elif "PIN_BAR" in signal_type or "ENGULFING" in signal_type:
                logger.debug(f"🚧 TRANSITION | Blocking {signal_type}")
            else:
                gated.append(signal)
        
        return gated
