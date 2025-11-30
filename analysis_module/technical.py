"""
Technical Analysis Module
Core calculations: PDH/PDL, Support/Resistance, Volume, Breakouts, Advanced TA
Includes debugging output for signal validation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from config.settings import (
    SR_CLUSTER_TOLERANCE,
    MIN_SR_TOUCHES,
    LOOKBACK_BARS,
    MIN_VOLUME_RATIO,
    VOLUME_PERIOD,
    FALSE_BREAKOUT_RETRACEMENT,
    RETEST_ZONE_PERCENT,
    MIN_RSI_BULLISH,
    MAX_RSI_BEARISH,
    RSI_PERIOD,
    ATR_PERIOD,
    EMA_SHORT,
    EMA_LONG,
    MIN_RISK_REWARD_RATIO,
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    DEBUG_MODE,
)

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types"""

    BULLISH_BREAKOUT = "BULLISH_BREAKOUT"
    BEARISH_BREAKOUT = "BEARISH_BREAKOUT"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    RETEST_SETUP = "RETEST_SETUP"
    INSIDE_BAR = "INSIDE_BAR"
    SUPPORT_BOUNCE = "SUPPORT_BOUNCE"
    RESISTANCE_BOUNCE = "RESISTANCE_BOUNCE"


@dataclass
class Signal:
    """Trading signal data class"""

    signal_type: SignalType
    instrument: str
    timeframe: str
    price_level: float
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float  # 0-100%
    volume_confirmed: bool
    momentum_confirmed: bool
    risk_reward_ratio: float
    timestamp: pd.Timestamp
    description: str
    debug_info: Dict = None


@dataclass
class TechnicalLevels:
    """Support/Resistance levels and related metrics"""

    support_levels: List[float]
    resistance_levels: List[float]
    pivot: float
    pdh: float
    pdl: float
    atr: float
    volatility_score: float


class TechnicalAnalyzer:
    """Main technical analysis engine"""

    def __init__(self, instrument: str):
        self.instrument = instrument
        logger.info(f"🔬 TechnicalAnalyzer initialized for {instrument}")

    # =====================================================================
    # PDH / PDL
    # =====================================================================

    def calculate_pdh_pdl(
        self, df: pd.DataFrame
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate Previous Day High & Low.

        Uses daily resample of df.
        """
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)

            daily_df = df.resample("D").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )

            if len(daily_df) < 2:
                logger.warning("❌ Not enough daily data for PDH/PDL")
                return None, None

            prev_day = daily_df.iloc[-2]
            pdh = float(prev_day["high"])
            pdl = float(prev_day["low"])

            current_open = float(daily_df.iloc[-1]["open"])
            gap_pct = (
                ((current_open - pdh) / pdh) * 100.0 if pdh > 0 else 0.0
            )

            logger.info(
                f"📊 PDH/PDL | PDH: {pdh:.2f} | PDL: {pdl:.2f} | "
                f"Gap% vs PDH(open): {gap_pct:.2f}%"
            )

            if DEBUG_MODE:
                logger.debug(
                    f"   PDH: {pdh:.2f}, PDL: {pdl:.2f}, "
                    f"Current open: {current_open:.2f}, Gap%: {gap_pct:.2f}"
                )

            return pdh, pdl

        except Exception as e:
            logger.error(f"❌ PDH/PDL calculation failed: {str(e)}")
            return None, None

    # =====================================================================
    # SUPPORT / RESISTANCE
    # =====================================================================

    def calculate_support_resistance(self, df: pd.DataFrame) -> TechnicalLevels:
        """
        Calculate Support/Resistance levels with clustering.
        """
        try:
            df = df.copy()
            lookback = min(LOOKBACK_BARS, len(df))
            df_sub = df.tail(lookback)

            highs = df_sub["high"].values
            lows = df_sub["low"].values

            support_levels = self._find_support_levels(lows)
            resistance_levels = self._find_resistance_levels(highs)

            support_clusters = self._cluster_levels(support_levels)
            resistance_clusters = self._cluster_levels(resistance_levels)

            pdh, pdl = self.calculate_pdh_pdl(df)
            pivot_std, _ = self._calculate_pivots(df_sub)

            atr = self._calculate_atr(df_sub)
            volatility_score = self._calculate_volatility_score(df_sub)

            logger.info(
                "✅ S/R Calculated | "
                f"Supports: {len(support_clusters)} | "
                f"Resistances: {len(resistance_clusters)}"
            )

            if DEBUG_MODE:
                logger.debug(
                    "   Supports: "
                    f"{[f'{s:.2f}' for s in support_clusters[:5]]}"
                )
                logger.debug(
                    "   Resistances: "
                    f"{[f'{r:.2f}' for r in resistance_clusters[:5]]}"
                )

            return TechnicalLevels(
                support_levels=sorted(support_clusters),
                resistance_levels=sorted(resistance_clusters, reverse=True),
                pivot=pivot_std,
                pdh=pdh or 0.0,
                pdl=pdl or 0.0,
                atr=atr,
                volatility_score=volatility_score,
            )

        except Exception as e:
            logger.error(f"❌ S/R calculation failed: {str(e)}")
            return TechnicalLevels([], [], 0.0, 0.0, 0.0, 0.0, 0.0)

    def _find_support_levels(self, lows: np.ndarray) -> List[float]:
        """Identify support levels via local minima."""
        supports: List[float] = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                supports.append(float(lows[i]))
        return supports

    def _find_resistance_levels(self, highs: np.ndarray) -> List[float]:
        """Identify resistance levels via local maxima."""
        resistances: List[float] = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                resistances.append(float(highs[i]))
        return resistances

    def _cluster_levels(
        self, levels: List[float], tolerance_pct: float = SR_CLUSTER_TOLERANCE
    ) -> List[float]:
        """Cluster nearby levels together."""
        if not levels:
            return []

        sorted_levels = sorted(levels)
        clusters: List[float] = []
        current_cluster: List[float] = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            cluster_mean = float(np.mean(current_cluster))
            tolerance = cluster_mean * (tolerance_pct / 100.0)

            if abs(level - cluster_mean) <= tolerance:
                current_cluster.append(level)
            else:
                clusters.append(float(np.mean(current_cluster)))
                current_cluster = [level]

        clusters.append(float(np.mean(current_cluster)))

        if DEBUG_MODE:
            logger.debug(
                f"   Clustered {len(sorted_levels)} into {len(clusters)} clusters"
            )

        return clusters

    def _calculate_pivots(
        self, df: pd.DataFrame
    ) -> Tuple[float, float]:
        """Calculate pivot points (Standard & simple Fibonacci-style)."""
        try:
            last_row = df.iloc[-1]
            high, low, close = (
                float(last_row["high"]),
                float(last_row["low"]),
                float(last_row["close"]),
            )

            pivot_std = (high + low + close) / 3.0
            pivot_fib = (high + low) / 2.0

            return pivot_std, pivot_fib

        except Exception as e:
            logger.warning(f"⚠️  Pivot calculation failed: {str(e)}")
            return 0.0, 0.0

    # =====================================================================
    # VOLUME ANALYSIS
    # =====================================================================

    def check_volume_confirmation(
        self, df: pd.DataFrame
    ) -> Tuple[bool, float, str]:
        """
        Check if current candle has volume confirmation.
        """
        try:
            df = df.copy()
            if len(df) < VOLUME_PERIOD + 2:
                return False, 0.0, ""

            current_vol = float(df["volume"].iloc[-1])
            avg_vol = float(
                df["volume"].iloc[-VOLUME_PERIOD - 1 : -1].mean()
            )
            ratio = current_vol / avg_vol if avg_vol > 0 else 0.0
            is_confirmed = ratio >= MIN_VOLUME_RATIO

            info = (
                f"Vol: {current_vol:.0f} | Avg: {avg_vol:.0f} | Ratio: {ratio:.2f}x"
            )

            if is_confirmed:
                logger.info(f"✅ Volume confirmed | {info}")
            else:
                logger.warning(f"⚠️  Low volume | {info}")

            if DEBUG_MODE:
                logger.debug(f"   Volume threshold: {MIN_VOLUME_RATIO}x")

            return is_confirmed, ratio, info

        except Exception as e:
            logger.error(f"❌ Volume check failed: {str(e)}")
            return False, 0.0, ""

    # =====================================================================
    # HIGHER TF CONTEXT (15m)
    # =====================================================================

    def get_higher_tf_context(self, df_15m: pd.DataFrame) -> Dict:
        """
        Build higher timeframe (15m) context:
        - Trend direction: UP / DOWN / FLAT
        - 15m RSI
        - 15m short/long EMAs
        """
        context = {
            "trend_direction": "FLAT",
            "ema_short_15": 0.0,
            "ema_long_15": 0.0,
            "rsi_15": 50.0,
        }

        try:
            if df_15m is None or df_15m.empty:
                logger.warning("⚠️  get_higher_tf_context: empty 15m data")
                return context

            df = df_15m.copy().sort_index()

            if len(df) < max(EMA_LONG, RSI_PERIOD) + 5:
                logger.warning(
                    "⚠️  get_higher_tf_context: not enough 15m bars"
                )
                return context

            ema_short = df["close"].ewm(
                span=EMA_SHORT, adjust=False
            ).mean()
            ema_long = df["close"].ewm(span=EMA_LONG, adjust=False).mean()

            ema_short_15 = float(ema_short.iloc[-1])
            ema_long_15 = float(ema_long.iloc[-1])
            rsi_15 = float(self._calculate_rsi(df))

            if ema_short_15 > ema_long_15 and rsi_15 >= 55:
                trend = "UP"
            elif ema_short_15 < ema_long_15 and rsi_15 <= 45:
                trend = "DOWN"
            else:
                trend = "FLAT"

            context.update(
                {
                    "trend_direction": trend,
                    "ema_short_15": ema_short_15,
                    "ema_long_15": ema_long_15,
                    "rsi_15": rsi_15,
                }
            )

            logger.info(
                "📐 Higher TF context | "
                f"Trend: {trend} | "
                f"EMA15s: {ema_short_15:.2f}/{ema_long_15:.2f} | "
                f"RSI15: {rsi_15:.1f}"
            )
            return context

        except Exception as e:
            logger.error(f"❌ get_higher_tf_context failed: {str(e)}")
            return context

    # =====================================================================
    # BREAKOUT DETECTION WITH MTF
    # =====================================================================

    def detect_breakout(
        self,
        df: pd.DataFrame,
        support_resistance: TechnicalLevels,
        higher_tf_context: Dict,
    ) -> Optional[Signal]:
        """
        Detect breakout above resistance or below support with
        multi-timeframe confirmation (15m trend + RSI).
        """
        try:
            if df is None or df.empty:
                return None

            df = df.copy()
            current = df.iloc[-1]
            current_price = float(current["close"])
            current_high = float(current["high"])
            current_low = float(current["low"])

            trend_dir = higher_tf_context.get("trend_direction", "FLAT")
            rsi_15 = float(higher_tf_context.get("rsi_15", 50.0))

            breakout_signal: Optional[Signal] = None

            rsi_5 = self._calculate_rsi(df)
            vol_confirmed, vol_ratio, _ = self.check_volume_confirmation(df)

            # ------------------------
            # Bullish breakout
            # ------------------------
            if support_resistance.resistance_levels:
                nearest_resistance = float(
                    support_resistance.resistance_levels[0]
                )

                if current_high > nearest_resistance:
                    mtf_ok = (
                        trend_dir == "UP"
                        and rsi_15 >= MIN_RSI_BULLISH
                    )

                    if not mtf_ok:
                        logger.info(
                            "⏭️  Bullish breakout ignored (MTF filter) | "
                            f"Trend15m: {trend_dir} | RSI15: {rsi_15:.1f}"
                        )
                    else:
                        logger.info(
                            "🚀 Bullish breakout candidate | "
                            f"Price: {current_price:.2f} > R: {nearest_resistance:.2f}"
                        )

                        atr = support_resistance.atr
                        sl = current_low - (atr * ATR_SL_MULTIPLIER)
                        tp = current_price + (atr * ATR_TP_MULTIPLIER)
                        risk_reward = (tp - current_price) / max(
                            current_price - sl, 1e-6
                        )

                        confidence = 60.0
                        if vol_confirmed:
                            confidence += 10
                        if rsi_5 >= MIN_RSI_BULLISH:
                            confidence += 10
                        if trend_dir == "UP":
                            confidence += 10
                        if risk_reward >= MIN_RISK_REWARD_RATIO:
                            confidence += 10

                        confidence = min(confidence, 95.0)

                        breakout_signal = Signal(
                            signal_type=SignalType.BULLISH_BREAKOUT,
                            instrument=self.instrument,
                            timeframe="5MIN",
                            price_level=nearest_resistance,
                            entry_price=current_price,
                            stop_loss=sl,
                            take_profit=tp,
                            confidence=confidence,
                            volume_confirmed=vol_confirmed,
                            momentum_confirmed=(
                                rsi_5 >= MIN_RSI_BULLISH
                            ),
                            risk_reward_ratio=risk_reward,
                            timestamp=pd.Timestamp.now(),
                            description=(
                                f"Bullish breakout at {nearest_resistance:.2f} | "
                                f"RR: {risk_reward:.2f} | "
                                f"MTF trend: {trend_dir}, RSI15: {rsi_15:.1f}"
                            ),
                            debug_info={
                                "volume_ratio": vol_ratio,
                                "rsi_5": rsi_5,
                                "rsi_15": rsi_15,
                                "trend_dir": trend_dir,
                                "atr": atr,
                            },
                        )

            # ------------------------
            # Bearish breakdown
            # ------------------------
            if support_resistance.support_levels:
                nearest_support = float(
                    support_resistance.support_levels[0]
                )

                if current_low < nearest_support:
                    mtf_ok = (
                        trend_dir == "DOWN"
                        and rsi_15 <= MAX_RSI_BEARISH
                    )

                    if not mtf_ok:
                        logger.info(
                            "⏭️  Bearish breakdown ignored (MTF filter) | "
                            f"Trend15m: {trend_dir} | RSI15: {rsi_15:.1f}"
                        )
                    else:
                        logger.info(
                            "📉 Bearish breakdown candidate | "
                            f"Price: {current_price:.2f} < S: {nearest_support:.2f}"
                        )

                        atr = support_resistance.atr
                        sl = current_high + (atr * ATR_SL_MULTIPLIER)
                        tp = current_price - (atr * ATR_TP_MULTIPLIER)
                        risk_reward = (current_price - tp) / max(
                            sl - current_price, 1e-6
                        )

                        confidence = 60.0
                        if vol_confirmed:
                            confidence += 10
                        if rsi_5 <= MAX_RSI_BEARISH:
                            confidence += 10
                        if trend_dir == "DOWN":
                            confidence += 10
                        if risk_reward >= MIN_RISK_REWARD_RATIO:
                            confidence += 10

                        confidence = min(confidence, 95.0)

                        breakout_signal = Signal(
                            signal_type=SignalType.BEARISH_BREAKOUT,
                            instrument=self.instrument,
                            timeframe="5MIN",
                            price_level=nearest_support,
                            entry_price=current_price,
                            stop_loss=sl,
                            take_profit=tp,
                            confidence=confidence,
                            volume_confirmed=vol_confirmed,
                            momentum_confirmed=(
                                rsi_5 <= MAX_RSI_BEARISH
                            ),
                            risk_reward_ratio=risk_reward,
                            timestamp=pd.Timestamp.now(),
                            description=(
                                f"Bearish breakdown at {nearest_support:.2f} | "
                                f"RR: {risk_reward:.2f} | "
                                f"MTF trend: {trend_dir}, RSI15: {rsi_15:.1f}"
                            ),
                            debug_info={
                                "volume_ratio": vol_ratio,
                                "rsi_5": rsi_5,
                                "rsi_15": rsi_15,
                                "trend_dir": trend_dir,
                                "atr": atr,
                            },
                        )

            return breakout_signal

        except Exception as e:
            logger.error(f"❌ Breakout detection failed: {str(e)}")
            return None

    # =====================================================================
    # FALSE BREAKOUT & RETEST
    # =====================================================================

    def detect_false_breakout(
        self,
        df: pd.DataFrame,
        breakout_level: float,
        direction: str,
    ) -> Tuple[bool, Dict]:
        """
        Detect false breakout (price fails to hold beyond level).

        Args:
            df: 5m OHLCV DataFrame
            breakout_level: breakout price level (support/resistance)
            direction: "UP" for bullish breakout, "DOWN" for bearish

        Returns:
            (is_false, details_dict)
        """
        details = {
            "retracement_pct": 0.0,
            "weak_volume": False,
            "rejection_candles": 0,
        }

        try:
            if df is None or df.empty:
                return False, details

            recent = df.tail(3).copy()
            if len(recent) < 2:
                return False, details

            vol_confirmed, vol_ratio, _ = self.check_volume_confirmation(df)
            weak_volume = not vol_confirmed

            false_breakout = False
            rejection_candles = 0
            retracement_pct = 0.0

            if direction.upper() == "UP":
                for idx in recent.index[1:]:
                    close_price = float(recent.loc[idx, "close"])
                    if close_price < breakout_level:
                        rejection_candles += 1
                        retracement_pct = (
                            (breakout_level - close_price)
                            / breakout_level
                            * 100.0
                        )

                if (
                    rejection_candles > 0
                    and retracement_pct >= FALSE_BREAKOUT_RETRACEMENT
                ):
                    false_breakout = True

            elif direction.upper() == "DOWN":
                for idx in recent.index[1:]:
                    close_price = float(recent.loc[idx, "close"])
                    if close_price > breakout_level:
                        rejection_candles += 1
                        retracement_pct = (
                            (close_price - breakout_level)
                            / breakout_level
                            * 100.0
                        )

                if (
                    rejection_candles > 0
                    and retracement_pct >= FALSE_BREAKOUT_RETRACEMENT
                ):
                    false_breakout = True

            if false_breakout and weak_volume:
                logger.warning(
                    "⚠️  FALSE BREAKOUT detected | "
                    f"Dir: {direction} | Level: {breakout_level:.2f} | "
                    f"Retrace: {retracement_pct:.2f}% | "
                    f"Vol weak (ratio: {vol_ratio:.2f}x)"
                )
            elif false_breakout:
                logger.warning(
                    "⚠️  FALSE BREAKOUT detected (price action) | "
                    f"Dir: {direction} | Level: {breakout_level:.2f} | "
                    f"Retrace: {retracement_pct:.2f}%"
                )

            details.update(
                {
                    "retracement_pct": retracement_pct,
                    "weak_volume": weak_volume,
                    "rejection_candles": rejection_candles,
                }
            )
            return false_breakout, details

        except Exception as e:
            logger.error(f"❌ False breakout detection failed: {str(e)}")
            return False, details

    def detect_retest_setup(
        self, df: pd.DataFrame, support_resistance: TechnicalLevels
    ) -> Optional[Signal]:
        """
        Detect retest setup (price returns to key level).
        """
        try:
            if df is None or df.empty:
                return None

            current = df.iloc[-1]
            current_price = float(current["close"])

            candidate_levels = (
                support_resistance.support_levels[:3]
                + support_resistance.resistance_levels[:3]
            )

            for level in candidate_levels:
                distance_pct = abs(current_price - level) / level * 100.0
                if distance_pct <= RETEST_ZONE_PERCENT:
                    logger.info(
                        "🎯 RETEST SETUP | "
                        f"Price: {current_price:.2f} | Level: {level:.2f} | "
                        f"Dist: {distance_pct:.3f}%"
                    )

                    atr = support_resistance.atr

                    if current_price < level:
                        signal_type = SignalType.SUPPORT_BOUNCE
                        description = f"Support retest at {level:.2f}"
                    else:
                        signal_type = SignalType.RESISTANCE_BOUNCE
                        description = f"Resistance retest at {level:.2f}"

                    return Signal(
                        signal_type=signal_type,
                        instrument=self.instrument,
                        timeframe="5MIN",
                        price_level=level,
                        entry_price=current_price,
                        stop_loss=current_price - atr,
                        take_profit=current_price + (atr * 2.0),
                        confidence=75.0,
                        volume_confirmed=False,
                        momentum_confirmed=False,
                        risk_reward_ratio=2.0,
                        timestamp=pd.Timestamp.now(),
                        description=description,
                        debug_info={"distance_pct": distance_pct},
                    )

            return None

        except Exception as e:
            logger.error(f"❌ Retest detection failed: {str(e)}")
            return None

    def detect_inside_bar(
        self, df: pd.DataFrame
    ) -> Optional[Signal]:
        """
        Detect inside bar setup (smaller candle inside previous candle).
        """
        try:
            if df is None or len(df) < 2:
                return None

            prev_candle = df.iloc[-2]
            curr_candle = df.iloc[-1]

            is_inside = (
                curr_candle["high"] < prev_candle["high"]
                and curr_candle["low"] > prev_candle["low"]
            )

            if is_inside:
                logger.info(
                    "📊 INSIDE BAR DETECTED | Setup ready for breakout trade"
                )

                atr = self._calculate_atr(df)

                return Signal(
                    signal_type=SignalType.INSIDE_BAR,
                    instrument=self.instrument,
                    timeframe="5MIN",
                    price_level=float(curr_candle["close"]),
                    entry_price=float(prev_candle["high"]),
                    stop_loss=float(prev_candle["low"]),
                    take_profit=float(prev_candle["high"]) + atr,
                    confidence=72.0,
                    volume_confirmed=False,
                    momentum_confirmed=False,
                    risk_reward_ratio=1.5,
                    timestamp=pd.Timestamp.now(),
                    description=(
                        "Inside bar setup - trade breakout of previous candle"
                    ),
                    debug_info={
                        "prev_high": float(prev_candle["high"]),
                        "prev_low": float(prev_candle["low"]),
                        "curr_high": float(curr_candle["high"]),
                        "curr_low": float(curr_candle["low"]),
                    },
                )

            return None

        except Exception as e:
            logger.error(f"❌ Inside bar detection failed: {str(e)}")
            return None

    # =====================================================================
    # INDICATOR CALCULATIONS
    # =====================================================================

    def _calculate_rsi(
        self, df: pd.DataFrame, period: int = RSI_PERIOD
    ) -> float:
        """Calculate RSI (Relative Strength Index)."""
        try:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0.0).rolling(period).mean()

            if loss.iloc[-1] == 0 or pd.isna(loss.iloc[-1]):
                return 50.0

            rs = gain / loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return float(rsi.iloc[-1])

        except Exception:
            return 50.0

    def _calculate_atr(
        self, df: pd.DataFrame, period: int = ATR_PERIOD
    ) -> float:
        """Calculate Average True Range."""
        try:
            high_low = df["high"] - df["low"]
            high_close = (df["high"] - df["close"].shift()).abs()
            low_close = (df["low"] - df["close"].shift()).abs()

            ranges = pd.concat(
                [high_low, high_close, low_close], axis=1
            )
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(period).mean()

            return float(atr.iloc[-1])

        except Exception:
            return 0.0

    def _calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """Calculate volatility score (0-100)."""
        try:
            returns = df["close"].pct_change()
            volatility = float(returns.std() * 100.0)
            score = min(volatility * 100.0, 100.0)
            return score
        except Exception:
            return 50.0

    # =====================================================================
    # TOP-LEVEL ANALYSIS WRAPPER (WITH MTF)
    # =====================================================================

    def analyze_with_multi_tf(
        self, df_5m: pd.DataFrame, higher_tf_context: Dict
    ) -> Dict:
        """
        Full analysis using 5m data + 15m higher timeframe context.
        """
        result = {
            "pdh": None,
            "pdl": None,
            "levels": None,
            "volume_confirmed": False,
            "volume_ratio": 0.0,
            "breakout_signal": None,
            "retest_signal": None,
            "inside_bar_signal": None,
        }

        try:
            if df_5m is None or df_5m.empty:
                logger.error("❌ analyze_with_multi_tf: empty 5m data")
                return result

            pdh, pdl = self.calculate_pdh_pdl(df_5m)
            levels = self.calculate_support_resistance(df_5m)
            vol_confirmed, vol_ratio, _ = self.check_volume_confirmation(df_5m)
            breakout = self.detect_breakout(df_5m, levels, higher_tf_context)
            retest = self.detect_retest_setup(df_5m, levels)
            inside_bar = self.detect_inside_bar(df_5m)

            result.update(
                {
                    "pdh": pdh,
                    "pdl": pdl,
                    "levels": levels,
                    "volume_confirmed": vol_confirmed,
                    "volume_ratio": vol_ratio,
                    "breakout_signal": breakout,
                    "retest_signal": retest,
                    "inside_bar_signal": inside_bar,
                }
            )
            return result

        except Exception as e:
            logger.error(f"❌ analyze_with_multi_tf failed: {str(e)}")
            return result


# =========================================================================
# HELPER
# =========================================================================


def analyze_instrument(df: pd.DataFrame, instrument: str) -> Dict:
    """
    Legacy helper (single timeframe).
    """
    analyzer = TechnicalAnalyzer(instrument)
    pdh, pdl = analyzer.calculate_pdh_pdl(df)
    levels = analyzer.calculate_support_resistance(df)
    vol_confirmed, vol_ratio, _ = analyzer.check_volume_confirmation(df)
    breakout = analyzer.detect_breakout(
        df, levels, {"trend_direction": "FLAT", "rsi_15": 50.0}
    )
    retest = analyzer.detect_retest_setup(df, levels)
    inside_bar = analyzer.detect_inside_bar(df)

    return {
        "pdh": pdh,
        "pdl": pdl,
        "levels": levels,
        "volume_confirmed": vol_confirmed,
        "volume_ratio": vol_ratio,
        "breakout_signal": breakout,
        "retest_signal": retest,
        "inside_bar_signal": inside_bar,
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("✅ Technical Analysis Module loaded")
