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
    DEBUG_MODE,
)

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch real-time market data with caching & error handling"""

    def __init__(self):
        # Simple placeholder base URL – you can swap this for a more robust NSE API
        self.base_url = "https://nse-api-khaki.vercel.app:5000"
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

        logger.info(f"🚀 DataFetcher initialized | Base URL: {self.base_url}")

    # =====================================================================
    # REAL-TIME DATA FETCHING
    # =====================================================================

    def fetch_nse_data(
        self, instrument: str, retries: int = 3
    ) -> Optional[Dict]:
        """
        Fetch real-time NSE data for an instrument.

        Args:
            instrument: 'NIFTY', 'BANKNIFTY', or 'FINNIFTY'
            retries: Number of retry attempts

        Returns:
            Dict with OHLCV data or None if failed
        """
        if instrument not in INSTRUMENTS:
            logger.error(f"❌ Unknown instrument: {instrument}")
            return None

        symbol = INSTRUMENTS[instrument]["symbol"]

        cache_key = f"nse_{instrument}"
        if self._is_cache_valid(cache_key, ttl=10):
            logger.debug(f"📦 Cache HIT: {instrument}")
            return self.cache[cache_key]

        logger.debug(f"📡 Fetching NSE data for: {instrument} ({symbol})")

        for attempt in range(retries):
            try:
                url = f"{self.base_url}/stock?symbol={symbol}"
                logger.debug(
                    f"   Attempt {attempt + 1}/{retries} | URL: {url}"
                )

                response = self.session.get(url, timeout=10)
                response.raise_for_status()

                data = response.json()

                if self._validate_nse_response(data):
                    logger.info(
                        f"✅ NSE data fetched: {instrument} | "
                        f"Price: {data.get('price', 'N/A')}"
                    )
                    self.cache[cache_key] = data
                    self.cache_time[cache_key] = time.time()

                    if DEBUG_MODE:
                        logger.debug(
                            "   Response sample: "
                            f"{json.dumps(data, indent=2)[:200]}..."
                        )

                    return data
                else:
                    logger.warning(
                        f"⚠️  Invalid NSE response for {instrument}"
                    )

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"⚠️  Attempt {attempt + 1} failed: {str(e)}"
                )
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"❌ Unexpected error: {str(e)}")

        logger.error(
            f"❌ Failed to fetch NSE data for {instrument} after {retries} attempts"
        )

        if cache_key in self.cache:
            logger.warning(
                f"⚠️  Falling back to cached data for {instrument}"
            )
            return self.cache[cache_key]

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

    def _validate_nse_response(self,  Dict) -> bool:
        """Validate NSE API response data quality."""
        required_fields = [
            "symbol",
            "price",
            "dayHigh",
            "dayLow",
            "volume",
            "timestamp",
        ]

        for field in required_fields:
            if field not in data:
                logger.warning(f"   ❌ Missing field: {field}")
                return False

        try:
            price = float(data.get("price"))
            volume = float(data.get("volume"))

            if price <= 0 or volume < 0:
                logger.warning(
                    f"   ❌ Invalid values | Price: {price} | Volume: {volume}"
                )
                return False

        except (ValueError, TypeError) as e:
            logger.warning(f"   ❌ Type conversion failed: {str(e)}")
            return False

        logger.debug("   ✅ Data validation passed")
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
                df = df.rename(columns={"Adj Close": "Close"})
            
            # Lowercase all column names
            df.columns = [str(c).lower() for c in df.columns]

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

        data = fetcher.fetch_nse_data(instrument)
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
