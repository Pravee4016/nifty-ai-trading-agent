"""
Data Fetching Module
Retrieves real-time OHLCV data and historical data.
Includes debugging, caching, and fallback mechanisms.
"""

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
import json
import os
import logging
from typing import Dict, List, Optional
import time

from config.settings import (
    CACHE_DIR,
    INSTRUMENTS,
    INSTRUMENTS,
    DEBUG_MODE,
    FYERS_CLIENT_ID,
)
from data_module.fyers_interface import FyersApp

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch real-time market data with caching & error handling"""

    def __init__(self):

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            }
        )
        self.cache: Dict[str, Dict] = {}
        self.cache_time: Dict[str, float] = {}

        # Initialize Fyers App
        self.fyers_app = FyersApp(app_id=FYERS_CLIENT_ID)
        
        logger.info(f"🚀 DataFetcher initialized | Primary: Fyers | Fallback: YFinance")

    # =====================================================================
    # REAL-TIME DATA FETCHING
    # =====================================================================

    def fetch_realtime_data(self, instrument: str) -> Optional[Dict]:
        """
        Fetch real-time market data from Fyers (Primary) or YFinance (Fallback).
        
        Returns:
            Dict: {
                'price': float,
                'dayHigh': float,
                'dayLow': float,
                'volume': float,
                'timestamp': str,
                'symbol': str
            }
        """
        if instrument not in INSTRUMENTS:
            logger.error(f"❌ Unknown instrument: {instrument}")
            return None

        cache_key = f"rt_{instrument}"
        if self._is_cache_valid(cache_key, ttl=10):
            return self.cache[cache_key]

        # 1. Try Fyers API
        data = self._fetch_fyers_data(instrument)
        if data:
            self.cache[cache_key] = data
            self.cache_time[cache_key] = time.time()
            return data

        # 2. Fallback to YFinance
        logger.warning(f"⚠️ Fyers failed, falling back to yfinance for {instrument}")
        data = self._fetch_yfinance_snapshot(instrument)
        if data:
            # Longer cache for yfinance as it's slower/delayed
            self.cache[cache_key] = data
            self.cache_time[cache_key] = time.time() 
            return data
            
        logger.error(f"❌ All data sources failed for {instrument}")
        return None

    def _fetch_fyers_data(self, instrument: str) -> Optional[Dict]:
        """Fetch from Fyers."""
        try:
            quote = self.fyers_app.get_quote(instrument)
            if not quote:
                return None
            
            # Debug Fyers structure
            if DEBUG_MODE:
                logger.info(f"DEBUG: Fyers Quote: {quote}")

            # Handle Fyers v3 structure: data might be in 'v' key
            # Check if 'lp' is directly in quote or in quote['v']
            data_source = quote.get('v', quote) if isinstance(quote.get('v'), dict) else quote
            
            # Map fields (Fyers v3 keys: lp, high_price, low_price, open_price, volume)
            # OR keys: lp, h, l, o, v (older/ws api?)
            # Let's handle both
            
            price = data_source.get('lp') or data_source.get('last_price') or 0
            high = data_source.get('high_price') or data_source.get('h') or 0
            low = data_source.get('low_price') or data_source.get('l') or 0
            open_p = data_source.get('open_price') or data_source.get('o') or 0
            volume = data_source.get('volume') or data_source.get('v') or 0

            return {
                "symbol": instrument,
                "price": float(price),
                "lastPrice": float(price),
                "dayHigh": float(high),
                "dayLow": float(low),
                "open": float(open_p),
                "volume": float(volume),
                "timestamp": datetime.now().isoformat(),
                "source": "FYERS"
            }
        except Exception as e:
            logger.error(f"❌ Fyers fetch error: {e}")
            return None

    def _fetch_yfinance_snapshot(self, instrument: str) -> Optional[Dict]:
        """Get snapshot from yfinance."""
        try:
            # Fetch 1 day of data with 5m interval to get latest candle
            df = self.fetch_historical_data(instrument, period="1d", interval="5m")
            if df is None or df.empty:
                return None
            
            # Preprocess to ensure standard lowercase columns
            df = self.preprocess_ohlcv(df)
                
            latest = df.iloc[-1]
            return {
                "symbol": instrument,
                "price": float(latest['close']),
                "dayHigh": float(df['high'].max()), 
                "dayLow": float(df['low'].min()),
                "open": float(df.iloc[0]['open']),
                "volume": float(df['volume'].sum()), 
                "timestamp": latest.name.isoformat(),
                "source": "YFINANCE"
            }
        except Exception as e:
            logger.error(f"❌ YFinance snapshot error: {e}")
            return None

    # =====================================================================
    # HISTORICAL DATA FETCHING (YFINANCE)
    # =====================================================================

    def fetch_historical_data(
        self,
        instrument: str,
        period: str = "5d",
        interval: str = "5m",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for technical analysis setup using yfinance.

        Args:
            instrument: 'NIFTY', 'BANKNIFTY', or 'FINNIFTY'
            period: '5d', '1mo', etc.
            interval: '5m', '15m', '1h', '1d'

        Returns:
            DataFrame with OHLCV or None
        """
        symbol_map = {
            "NIFTY": "^NSEI",
            "BANKNIFTY": "^NSEBANK",
            "FINNIFTY": "^NSEINFRA",  # adjust as needed
        }

        if instrument not in symbol_map:
            logger.error(f"❌ No yfinance symbol for instrument: {instrument}")
            return None

        yf_symbol = symbol_map[instrument]

        logger.debug(
            f"📊 Fetching historical data | "
            f"Instrument: {instrument} | Period: {period} | Interval: {interval}"
        )

        try:
            df = yf.download(
                yf_symbol,
                period=period,
                interval=interval,
                progress=False,
                prepost=False,
                auto_adjust=True,  # Fix FutureWarning & ensure consistent Close
            )

            if isinstance(df, pd.Series):
                df = df.to_frame().T

            if df.empty:
                logger.warning(
                    f"⚠️  Empty historical data returned for {instrument}"
                )
                return None

            logger.info(
                f"✅ Historical data fetched: {instrument} | {len(df)} candles"
            )
            return df

        except Exception as e:
            logger.error(f"❌ Failed to fetch historical  {str(e)}")
            return None

    def get_historical_data(self, instrument: str, interval: str, bars: int) -> Optional[pd.DataFrame]:
        """
        Get historical data with bars-based signature (for choppy session filter).
        
        Args:
            instrument: 'NIFTY', 'BANKNIFTY', or 'FINNIFTY'
            interval: '5m', '15m', etc
            bars: Number of bars needed
            
        Returns:
            DataFrame with OHLCV or None
        """
        # Convert bars to period
        # Rough estimate: 78 5-min bars per trading day
        if interval == "5m":
            days = max(1, (bars // 78) + 1)
        elif interval == "15m":
            days = max(1, (bars // 26) + 1)
        elif interval == "1h":
            days = max(2, (bars // 6) + 1)
        else:
            days = 5
        
        period = f"{days}d"
        df = self.fetch_historical_data(instrument, period, interval)
        
        if df is not None and not df.empty:
            # Return last N bars
            return df.tail(bars)
        return df

    def get_previous_day_stats(self, instrument: str) -> Optional[Dict]:
        """
        Fetch Previous Day High (PDH) and Low (PDL).
        """
        try:
            df = self.fetch_historical_data(instrument, period="5d", interval="1d")
            if df is None or len(df) < 2:
                return None
            
            # Get previous day (last completed candle)
            prev_day = df.iloc[-2]
            
            return {
                "pdh": prev_day["high"],
                "pdl": prev_day["low"],
                "pdc": prev_day["close"],
                "date": prev_day.name.strftime("%Y-%m-%d")
            }
        except Exception as e:
            logger.error(f"❌ Failed to fetch PDH/PDL for {instrument}: {str(e)}")
            return None

    def get_opening_range_stats(self, instrument: str) -> Optional[Dict]:
        """
        Fetch High/Low of the first 5-min and 15-min candles of the current day.
        """
        try:
            # Fetch today's data (1d period, 5m interval)
            df_5m = self.fetch_historical_data(instrument, period="1d", interval="5m")
            df_15m = self.fetch_historical_data(instrument, period="1d", interval="15m")
            
            stats = {}
            
            if df_5m is not None and not df_5m.empty:
                first_5m = df_5m.iloc[0]
                stats["orb_5m_high"] = first_5m["high"]
                stats["orb_5m_low"] = first_5m["low"]
                
            if df_15m is not None and not df_15m.empty:
                first_15m = df_15m.iloc[0]
                stats["orb_15m_high"] = first_15m["high"]
                stats["orb_15m_low"] = first_15m["low"]
                
            return stats if stats else None
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch Opening Range for {instrument}: {str(e)}")
            return None

    # =====================================================================
    # DATA VALIDATION
    # =====================================================================

    def _validate_nse_response(self, data: Dict) -> bool:
        """Deprecated validation."""
        return True

    # =====================================================================
    # CACHING
    # =====================================================================

    def _is_cache_valid(self, cache_key: str, ttl: int = 60) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_time:
            return False

        age = time.time() - self.cache_time[cache_key]
        is_valid = age < ttl

        if DEBUG_MODE:
            logger.debug(
                f"   Cache age: {age:.1f}s | TTL: {ttl}s | Valid: {is_valid}"
            )

        return is_valid

    def clear_cache(self, instrument: Optional[str] = None):
        """Clear cache for specific instrument or all."""
        if instrument:
            cache_key = f"nse_{instrument}"
            if cache_key in self.cache:
                del self.cache[cache_key]
                del self.cache_time[cache_key]
                logger.info(f"🗑️  Cleared cache for {instrument}")
        else:
            self.cache.clear()
            self.cache_time.clear()
            logger.info("🗑️  Cleared all cache")

    # =====================================================================
    # PREPROCESSING
    # =====================================================================

    def preprocess_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess OHLCV data for analysis.

        Adds:
        - hl_range
        - hl_mid
        - co_change
        - typical_price
        """
        try:
            df = df.copy()
            
            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                # Flatten multi-index columns (e.g., ('Close', 'NIFTY') -> 'Close')
                df.columns = df.columns.get_level_values(0)
            
            # Rename 'Adj Close' to 'Close' if 'Close' is missing
            if "Close" in df.columns and "Adj Close" in df.columns:
                # If both exist, drop 'Adj Close' and keep 'Close'
                df = df.drop(columns=["Adj Close"])
            elif "Adj Close" in df.columns:
                # If only 'Adj Close' exists, rename it to 'Close'
                # If only 'Adj Close' exists, rename it to 'Close'
                df = df.rename(columns={"Adj Close": "Close"})
            
            # Ensure DatetimeIndex (Robustness Fix)
            # If index is just numbers but we have a 'Date' or 'Datetime' column, use it.
            if not isinstance(df.index, pd.DatetimeIndex):
                for time_col in ["Date", "date", "Datetime", "datetime", "timestamp"]:
                    if time_col in df.columns:
                        df[time_col] = pd.to_datetime(df[time_col])
                        df.set_index(time_col, inplace=True)
                        logger.debug(f"   ✅ Set DatetimeIndex from column: {time_col}")
                        break
            
            # Lowercase all column names
            df.columns = [str(c).lower() for c in df.columns]
            
            if DEBUG_MODE:
                logger.debug(f"   Columns after processing: {list(df.columns)}")

            if not {"open", "high", "low", "close"}.issubset(df.columns):
                logger.warning(f"⚠️  preprocess_ohlcv: missing OHLC columns. Have: {list(df.columns)}")
                return df

            df["hl_range"] = df["high"] - df["low"]
            df["hl_mid"] = (df["high"] + df["low"]) / 2.0
            df["co_change"] = (df["close"] - df["open"]) / df["open"] * 100.0
            df["typical_price"] = (
                df["high"] + df["low"] + df["close"]
            ) / 3.0

            logger.debug(
                f"✅ Preprocessed {len(df)} candles | Added calculated fields"
            )
            return df

        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {str(e)}")
            return df

    # =====================================================================
    # SAVE / LOAD CSV (for debugging/backtesting)
    # =====================================================================

    def save_to_csv(
        self, df: pd.DataFrame, instrument: str, filename: Optional[str] = None
    ):
        """Save data to CSV for offline analysis."""
        try:
            if filename is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{CACHE_DIR}/{instrument}_{ts}.csv"

            df.to_csv(filename)
            logger.info(f"💾 Saved  {filename}")

        except Exception as e:
            logger.error(f"❌ Failed to save CSV: {str(e)}")

    def load_from_csv(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from CSV."""
        try:
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            logger.info(
                f"📂 Loaded  {filename} | {len(df)} rows"
            )
            return df

        except Exception as e:
            logger.error(f"❌ Failed to load CSV: {str(e)}")
            return None


# =========================================================================
# UTILITY
# =========================================================================


_fetcher: Optional[DataFetcher] = None


def get_data_fetcher() -> DataFetcher:
    """Singleton pattern for DataFetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = DataFetcher()
    return _fetcher


def test_data_fetcher():
    """Test the data fetcher with NIFTY and BANKNIFTY."""
    logger.info("🧪 Testing DataFetcher...")

    fetcher = get_data_fetcher()

    for instrument in ["NIFTY", "BANKNIFTY"]:
        logger.info(f"\n📌 Testing {instrument}...")

        data = fetcher.fetch_realtime_data(instrument)
        if data is not None:
            logger.info(f"   ✅ Real-time: {data.get('price')}")

        df = fetcher.fetch_historical_data(
            instrument, period="5d", interval="5m"
        )
        if df is not None:
            logger.info(f"   ✅ Historical: {len(df)} candles")

    logger.info("\n✅ DataFetcher tests completed!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if DEBUG_MODE else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    test_data_fetcher()
