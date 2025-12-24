# NIFTY AI Trading Agent - Complete Project Code

Generated: 2025-12-20

This document contains the complete codebase of the NIFTY AI Trading Agent.

---

## Table of Contents

### Configuration Files
- [requirements.txt](#requirementstxt)
- [Dockerfile](#Dockerfile)
- [.gcloudignore](#gcloudignore)
- [deploy.sh](#deploysh)
- [deploy_job.sh](#deploy_jobsh)
- [schedule_job.sh](#schedule_jobsh)

### Core Application
- [./app/__init__.py](#-app-__init__py)
- [./app/agent.py](#-app-agentpy)
- [./app/bootstrap.py](#-app-bootstrappy)
- [./app/selfcheck.py](#-app-selfcheckpy)
- [./main.py](#-mainpy)

### Configuration Module
- [./config/__init__.py](#-config-__init__py)
- [./config/logging_config.py](#-config-logging_configpy)
- [./config/settings.py](#-config-settingspy)

### Data Module
- [./data_module/__init__.py](#-data_module-__init__py)
- [./data_module/fetcher.py](#-data_module-fetcherpy)
- [./data_module/fyers_interface.py](#-data_module-fyers_interfacepy)
- [./data_module/fyers_oauth.py](#-data_module-fyers_oauthpy)
- [./data_module/ml_data_collector.py](#-data_module-ml_data_collectorpy)
- [./data_module/option_chain_fetcher.py](#-data_module-option_chain_fetcherpy)
- [./data_module/persistence.py](#-data_module-persistencepy)
- [./data_module/persistence_models.py](#-data_module-persistence_modelspy)
- [./data_module/trade_tracker.py](#-data_module-trade_trackerpy)

### Analysis Module
- [./analysis_module/__init__.py](#-analysis_module-__init__py)
- [./analysis_module/adaptive_thresholds.py](#-analysis_module-adaptive_thresholdspy)
- [./analysis_module/combo_signals.py](#-analysis_module-combo_signalspy)
- [./analysis_module/confluence_detector.py](#-analysis_module-confluence_detectorpy)
- [./analysis_module/manipulation_guard.py](#-analysis_module-manipulation_guardpy)
- [./analysis_module/market_state_engine.py](#-analysis_module-market_state_enginepy)
- [./analysis_module/option_chain_analyzer.py](#-analysis_module-option_chain_analyzerpy)
- [./analysis_module/signal_pipeline.py](#-analysis_module-signal_pipelinepy)
- [./analysis_module/technical.py](#-analysis_module-technicalpy)

### ML Module
- [./ml_module/__init__.py](#-ml_module-__init__py)
- [./ml_module/feature_extractor.py](#-ml_module-feature_extractorpy)
- [./ml_module/model_storage.py](#-ml_module-model_storagepy)
- [./ml_module/predictor.py](#-ml_module-predictorpy)

### AI Module
- [./ai_module/ai_factory.py](#-ai_module-ai_factorypy)
- [./ai_module/groq_analyzer.py](#-ai_module-groq_analyzerpy)
- [./ai_module/vertex_analyzer.py](#-ai_module-vertex_analyzerpy)

### Telegram Module
- [./telegram_module/bot_handler.py](#-telegram_module-bot_handlerpy)

### Tests
- [./tests/__init__.py](#-tests-__init__py)
- [./tests/test_adaptive_thresholds.py](#-tests-test_adaptive_thresholdspy)
- [./tests/test_option_chain.py](#-tests-test_option_chainpy)
- [./tests/test_risk_management.py](#-tests-test_risk_managementpy)
- [./tests/test_scoring.py](#-tests-test_scoringpy)

---

## Configuration Files

### requirements.txt

```text
# Core
pandas>=2.0.3
numpy>=1.24.3
python-dateutil==2.8.2
pytz==2023.3
python-dotenv

# HTTP / API
requests==2.31.0

# Config
python-dotenv==1.0.0

# Market data
yfinance>=0.2.40

# Scheduling (optional for local scheduler)
APScheduler==3.10.4

# Telegram
python-telegram-bot==20.3

# Timezone helpers
tzdata==2023.3
google-cloud-firestore
fyers-apiv3>=3.0.0
google-cloud-secret-manager>=2.16.0

# Machine Learning
lightgbm>=4.1.0
scikit-learn>=1.3.0
google-cloud-storage>=2.10.0  # For GCS model storage
google-cloud-aiplatform>=1.38.0  # For Vertex AI Gemini

```

### Dockerfile

```dockerfile
# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for Google credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/trading-agent-key.json"

ENV GOOGLE_APPLICATION_CREDENTIALS="/app/trading-agent-key.json"

# Run main bot
CMD ["python", "main.py"]
```

### .gcloudignore

```text
# Exclude venv
venv/
__pycache__/
*.pyc

# Exclude editor files
.vscode/
.idea/

# Exclude Google SDK downloaded to project (you should NOT keep it here)
google-cloud-sdk/

# Exclude logs
*.log
logs/

# Exclude tar archives
*.tar.gz
*.zip

# Exclude system junk
.DS_Store
```

### deploy.sh

```bash
#!/bin/bash
#
# Deployment script for Nifty AI Trading Agent
# Deploys to Google Cloud Function (Gen 2)
#

set -e

# Configuration
# Use env var if set, otherwise use gcloud config
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
else
    PROJECT_ID="${GOOGLE_CLOUD_PROJECT}"
fi

FUNCTION_NAME="nifty-trading-agent"
REGION="asia-south1"
RUNTIME="python311"
ENTRY_POINT="main"
MEMORY="512MB"
TIMEOUT="540s"

echo "===================================="
echo "Nifty AI Trading Agent - Deployment"
echo "===================================="
echo ""

# Check if project ID is set
if [ -z "$PROJECT_ID" ]; then
    echo "❌ GOOGLE_CLOUD_PROJECT environment variable not set"
    echo "Please set it: export GOOGLE_CLOUD_PROJECT=your-project-id"
    exit 1
fi

echo "📦 Project: $PROJECT_ID"
echo "🌏 Region: $REGION"
echo "⚡ Function: $FUNCTION_NAME"
echo ""

# Run tests first
echo "🧪 Running syntax checks..."
python3 -m py_compile analysis_module/market_state_engine.py
python3 -m py_compile analysis_module/signal_pipeline.py
python3 -m py_compile app/agent.py

echo "✅ Syntax checks passed"
echo ""

# Deploy
echo "🚀 Deploying to Cloud Functions..."
gcloud functions deploy $FUNCTION_NAME \
  --gen2 \
  --region=$REGION \
  --runtime=$RUNTIME \
  --entry-point=$ENTRY_POINT \
  --trigger-http \
  --allow-unauthenticated \
  --memory=$MEMORY \
  --timeout=$TIMEOUT \
  --set-env-vars="USE_ML_FILTERING=False" \
  --source=.

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Monitor logs:"
echo "   gcloud functions logs read $FUNCTION_NAME --region=$REGION --limit=50"
echo ""
echo "🔍 Test function:"
echo "   curl \$(gcloud functions describe $FUNCTION_NAME --region=$REGION --format='value(serviceConfig.uri)')"
echo ""

```

### deploy_job.sh

```bash
#!/bin/bash
set -e

# Deploy updated code to Cloud Run Job
# This updates the container image that the scheduler triggers

PROJECT_ID="nifty-trading-agent"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/trading-agent:latest"
JOB_NAME="trading-agent-job"

echo "🔨 Building and Pushing Docker image via Cloud Build..."
gcloud builds submit --tag $IMAGE_NAME .

echo "🚀 Updating Cloud Run Job..."
# Note: Don't use --env-vars-file as it conflicts with secret references already set
gcloud run jobs update $JOB_NAME \
    --region=$REGION \
    --image=$IMAGE_NAME

echo "🔧 Patching DEPLOYMENT_MODE..."
gcloud run jobs update $JOB_NAME \
    --region=$REGION \
    --update-env-vars="DEPLOYMENT_MODE=GCP"

echo "✅ Cloud Run Job updated successfully!"
echo ""
echo "📊 Job details:"
gcloud run jobs describe $JOB_NAME --region=$REGION --format="yaml(metadata.name,status.latestCreatedExecution)"

```

### schedule_job.sh

```bash
#!/bin/bash
set -e

# Schedule the Cloud Run Job
# Runs every 5 minutes from 09:15 to 15:30 IST (Mon-Fri)

JOB_NAME="trading-agent-job"
SCHEDULER_NAME="trading-agent-scheduler"
REGION="us-central1"
PROJECT_ID="nifty-trading-agent"
SERVICE_ACCOUNT="499697087516-compute@developer.gserviceaccount.com"

# API Endpoint for executing the job
URI="https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/$JOB_NAME:run"

echo "🔄 Updating Cloud Scheduler to target Cloud Run Job..."

# Delete existing if it exists (simplest way to ensure clean switch from Function to Job)
if gcloud scheduler jobs describe $SCHEDULER_NAME --location=$REGION >/dev/null 2>&1; then
    gcloud scheduler jobs delete $SCHEDULER_NAME --location=$REGION --quiet
fi

# Create new scheduler job
gcloud scheduler jobs create http $SCHEDULER_NAME \
    --location=$REGION \
    --schedule="*/5 3-10 * * 1-5" \
    --time-zone="Etc/UTC" \
    --uri="$URI" \
    --http-method=POST \
    --oauth-service-account-email="$SERVICE_ACCOUNT" \
    --headers="Content-Type=application/json" \
    --message-body='{"overrides": {}}' # Empty overrides

echo "✅ Scheduler updated to target Cloud Run Job: $JOB_NAME"

```

## Core Application

### ./app/__init__.py

```python

```

### ./app/agent.py

```python
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
from ai_module.ai_factory import get_default_analyzer  # AI provider abstraction (Groq/Vertex/Hybrid)
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
        self.ai_analyzer = get_default_analyzer()  # Uses AI_PROVIDER from settings (Groq/Vertex/Hybrid)
        self.persistence = get_persistence()
        self.option_fetcher = OptionChainFetcher()
        self.circuit_breaker = CircuitBreaker()
        
        # Initialize ML Data Collector (if USE_ML_FILTERING enabled)
        self.ml_data_collector = None
        from config.settings import USE_ML_FILTERING
        if USE_ML_FILTERING:
            try:
                from data_module.ml_data_collector import MLDataCollector
                from data_module.persistence import get_firestore_client
                db = get_firestore_client()
                self.ml_data_collector = MLDataCollector(db)
                logger.info("✅ ML data collector initialized")
            except Exception as e:
                logger.warning(f"⚠️ ML data collector failed: {e}")
        
        # Initialize Signal Pipeline
        self.signal_pipeline = SignalPipeline(groq_analyzer=self.ai_analyzer)  # ai_analyzer works with any provider
        
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

            # --- FETCH INDIA VIX (Phase 3) ---
            india_vix = self.fetcher.fetch_india_vix()
            if india_vix:
                logger.info(f"📊 India VIX: {india_vix:.2f}")

            analyzer = TechnicalAnalyzer(instrument)
            higher_tf_context = analyzer.get_higher_tf_context(df_15m, df_5m, df_daily, india_vix=india_vix)
            
            # Add VIX to context for adaptive thresholds
            higher_tf_context["india_vix"] = india_vix
            
            # Update global market context for AI
            self.market_context[instrument] = {
                "trend_5m": higher_tf_context.get("trend_5m", "NEUTRAL"),
                "trend_15m": higher_tf_context.get("trend_15m", "NEUTRAL"),
                "trend_daily": higher_tf_context.get("trend_daily", "NEUTRAL"),
                "rsi_15": higher_tf_context.get("rsi_15", 50.0),
                "rsi_long_threshold": higher_tf_context.get("rsi_long_threshold", 60.0),
                "rsi_short_threshold": higher_tf_context.get("rsi_short_threshold", 40.0),
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
            # Add dataframe and VWAP to analysis for Market State Engine
            analysis["df"] = df_5m  # OHLCV dataframe for state evaluation
            analysis["vwap_series"] = df_5m['VWAP'] if 'VWAP' in df_5m.columns else None
            
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
                        time_diff = abs((now - timestamp).total_seconds()) / 60.0
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
                            conflict_diff = abs((now - timestamp).total_seconds()) / 60.0
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
                # Enhanced error logging
                logger.error("   ❌ Telegram alert failed - investigating cause...")
                logger.error(f"   Signal Type: {stype}")
                logger.error(f"   Instrument: {instrument}")
                logger.error(f"   Level: {price_level:.2f}")
                
                # Test connection to see if bot is still reachable
                try:
                    if not self.telegram_bot.test_connection():
                        logger.error("   ❌ Telegram bot connection lost - credentials may be invalid")
                    else:
                        logger.error("   ⚠️  Telegram connection OK but message send failed - check telegram_module logs for details")
                except Exception as conn_err:
                    logger.error(f"   ❌ Cannot test Telegram connection: {conn_err}", exc_info=True)
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
                "stats": {  # Changed from 'statistics' to 'stats' to match bot_handler.py line 453
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
            
            # Get performance stats (Today only)
            summary["performance"] = self.trade_tracker.get_stats(days=0)
            
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



```

### ./app/bootstrap.py

```python
"""
Bootstrap Module
Handles the creation and initialization of the Trading Agent.
Implements the AppFactory pattern.
"""
import logging
from app.agent import NiftyTradingAgent
from config.logging_config import setup_logging

logger = setup_logging(__name__)

def create_agent() -> NiftyTradingAgent:
    """
    Factory function to create and configure the trading agent.
    """
    try:
        logger.info("🔧 Bootstrapping NiftyTradingAgent...")
        agent = NiftyTradingAgent()
        return agent
    except Exception as e:
        logger.critical(f"❌ Failed to create agent: {e}", exc_info=True)
        raise

```

### ./app/selfcheck.py

```python
"""
Self-Check Module
Verifies environment, configuration, and API connectivity.
"""
import logging
import os
import sys

# Ensure project root is in sys.path to allow imports from config/ and app/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import validate_config, LOG_DIR, CACHE_DIR
from config.logging_config import setup_logging
from app.bootstrap import create_agent

logger = setup_logging(__name__)

def check_directories():
    """Verify write access to critical directories."""
    dirs = [LOG_DIR, CACHE_DIR]
    success = True
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            test_file = os.path.join(d, "test_write.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"✅ Write access confirmed: {d}")
        except Exception as e:
            logger.error(f"❌ Write access FAILED: {d} | {e}")
            success = False
    return success

def run_self_check():
    """Run full system check."""
    logger.info("🔍 Starting System Self-Check...")
    
    # 1. Config Validation
    errors = validate_config()
    if errors:
        for e in errors:
            logger.error(e)
        logger.error("❌ Configuration Validation FAILED")
        return False
    logger.info("✅ Configuration Validated")
    
    # 2. Directory Access
    if not check_directories():
        return False

    # 3. Agent Initialization & Connectivity
    try:
        agent = create_agent()
        
        # Grid Connectivity
        groq_ok = agent.ai_analyzer.test_connection()
        if groq_ok:
            logger.info("✅ Groq API Connected")
        else:
            logger.error("❌ Groq API Connection FAILED")
            
        tg_ok = agent.telegram_bot.test_connection()
        if tg_ok:
            logger.info("✅ Telegram Bot Connected")
        else:
            logger.error("❌ Telegram Bot Connection FAILED")
            
        if groq_ok and tg_ok:
            logger.info("✅ All Systems Operational")
            return True
        else:
            return False
            
    except Exception as e:
        logger.critical(f"❌ Agent Bootstrap Failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_self_check()
    exit(0 if success else 1)

```

### ./main.py

```python
"""
Nifty AI Trading Agent - Main Entry Point
Uses AppFactory pattern and centralized logging.
"""
import logging
import pytz
from datetime import datetime, time
import argparse

# Config
from config.settings import TIME_ZONE, DEBUG_MODE
from config.logging_config import setup_logging

# App Bootstrap
from app.bootstrap import create_agent

# Dependencies for standalone tasks
from data_module.persistence import get_persistence
from telegram_module.bot_handler import get_bot

# Setup Logger
logger = setup_logging(__name__)

# ------------------------------------------------------
# Helper Functions
# ------------------------------------------------------

def _is_market_hours_quick():
    """Quick market hours check for early exit (Cloud Function optimization)."""
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).time()
    today = datetime.now(ist).weekday()
    # Allow execution until 16:30 for EOD activities (Summary, Market Closed Alert)
    # Weekdays only (0-4)
    if today > 4: return False
    return time(9, 15) <= now <= time(16, 30)

def check_and_send_market_closed_alert():
    """
    Check if 'Market Closed' alert has been sent today. 
    If not, and it is after 15:30 IST, send it once.
    """
    try:
        tz = pytz.timezone(TIME_ZONE)
        now = datetime.now(tz)
        
        # Only check if it's after market close (15:30)
        # We use 15:30 inclusive in case the job runs exactly then
        if now.time() >= time(15, 30):
            persistence = get_persistence()
            stats = persistence.get_daily_stats()
            
            if not stats.get("market_closed_msg_sent"):
                logger.info("🌙 Market Closed - Sending one-time alert")
                bot = get_bot()
                bot.send_message("🌙 <b>Market Closed</b> - Analysis Paused")
                persistence.increment_stat("market_closed_msg_sent")
            else:
                logger.debug("🌙 Market Closed - Alert already sent today")
                
    except Exception as e:
        logger.error(f"❌ Failed to check/send market closed alert: {e}")

# ------------------------------------------------------
# Main Execution
# ------------------------------------------------------

def main():
    """Main entry point for local run or cloud function."""
    logger.info("🚀 Agent starting...")
    
    # Use Factory to create agent
    try:
        agent = create_agent()
    except Exception as e:
        logger.critical("🔥 Agent initialization failed. Exiting.")
        return

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

    # Check for scheduled messages (Startup, Market Context, EOD Summary)
    # Must run outside is_market_hours() to catch EOD summary at 15:31+
    agent.check_scheduled_messages()

    if agent.is_market_hours():
        # Run Analysis
        results = agent.run_analysis()
    else:
        logger.info("⏰ Outside market hours - skipping analysis")
        check_and_send_market_closed_alert()

    stats = agent.get_statistics()
    logger.info("\n📈 STATISTICS")
    logger.info(f"Alerts Sent: {stats['alerts_sent']}")
    logger.info(f"ML Predictions: {stats.get('ml_predictions', 0)}")
    
    # Handle AI usage stats (supports both old Groq format and new factory format)
    ai_usage = stats.get('ai_usage', {})
    if ai_usage:
        # Check if it's the new hybrid/factory format
        if 'mode' in ai_usage:
            logger.info(f" AI Mode: {ai_usage['mode']}")
        # Try to get tokens used (may not exist for all providers)
        tokens = ai_usage.get('tokens_used', 'N/A')
        if tokens != 'N/A':
            logger.info(f" AI Usage: {tokens} tokens used")
    
    logger.info(f"Session Time: {stats.get('execution_time', 0):.2f}s")


def cloud_function_handler(request):
    """Entry point for Google Cloud Functions."""
    # Early exit for outside market hours (avoid full initialization)
    if not _is_market_hours_quick():
        # Check if we need to send the "Market Closed" alert before exiting
        check_and_send_market_closed_alert()
        return {"status": "skipped", "message": "Outside market hours"}
    
    logger.info("☁️  Cloud Function triggered")
    main()
    return {"status": "success", "message": "Analysis completed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nifty AI Trading Agent")
    parser.add_argument(
        "--once", action="store_true", help="Run analysis once"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test connections only"
    )

    args = parser.parse_args()

    if args.test:
        logger.info("🔧 Running Connection Tests...")
        try:
            agent = create_agent()
            logger.info("Testing connections...")
            if agent.ai_analyzer.test_connection():
                logger.info("✅ AI Connection OK")
            else:
                logger.error("❌ AI Connection FAILED")
                
            if agent.telegram_bot.test_connection():
                 logger.info("✅ Telegram Connection OK")
            else:
                 logger.error("❌ Telegram Connection FAILED")
                 
        except Exception as e:
            logger.error(f"❌ Test initialization failed: {e}")
    else:
        main()

```

## Configuration

### ./config/__init__.py

```python

```

### ./config/logging_config.py

```python
"""
Centralized Logging Configuration
"""
import logging
import os
import sys
from config.settings import LOG_DIR, DEBUG_MODE

def setup_logging(name: str = __name__) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.
    """
    log_format = "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
    log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
    
    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "trading_agent.log")
    
    # Get the root logger or custom logger
    logger = logging.getLogger() # Root logger
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.setLevel(log_level)
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)
        
        # File Handler
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    # Return logger for the specific module
    return logging.getLogger(name)

```

### ./config/settings.py

```python
"""
Configuration & Settings Module
All API keys, thresholds, and constants in one place for easy debugging
"""

import os
from dotenv import load_dotenv
from enum import Enum

# Load environment variables
load_dotenv()

# ============================================================================
# API KEYS & CREDENTIALS
# ============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_KEY_HERE")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID_HERE")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID") # Optional Channel ID

# ============================================================================
# FYERS CONFIGURATION
# ============================================================================
FYERS_CLIENT_ID = os.getenv("FYERS_CLIENT_ID", "DURQKS8D17-100") # Default from codebase
FYERS_SECRET_ID = os.getenv("FYERS_SECRET_ID")  # For OAuth refresh token support
FYERS_ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")


# ============================================================================
# MARKET PARAMETERS
# ============================================================================


class Instrument(Enum):
    """Trading instruments"""

    NIFTY50 = "NIFTY 50"
    BANKNIFTY = "BANKNIFTY"
    FINNIFTY = "FINNIFTY"


INSTRUMENTS = {
    "NIFTY": {
        "symbol": "NIFTY 50",
        "display_name": "Nifty 50",
        "tick_size": 0.05,  # Legacy - use get_tick_size() instead
        "lot_size": 75,
        "active": True,
    },
    "BANKNIFTY": {
        "symbol": "BANKNIFTY",
        "display_name": "Bank Nifty",
        "tick_size": 0.05,  # Legacy - use get_tick_size() instead
        "lot_size": 15,
        "active": False,  # DISABLED - Focus on NIFTY only
    },
    "FINNIFTY": {
        "symbol": "FINNIFTY",
        "display_name": "Fin Nifty",
        "tick_size": 0.05,  # Legacy - use get_tick_size() instead
        "lot_size": 40,
        "active": False,  # Enable when ready
    },
}

# ============================================================================
# TICK SIZE MAPPING (Spot vs Options)
# ============================================================================

# CRITICAL: Spot instruments trade in 1-point increments, NOT 0.05
# 0.05 tick size is ONLY for options (CE/PE)
TICK_SIZE_MAP = {
    "NIFTY_SPOT": 1.0,           # NIFTY spot/futures move in 1.0 point increments
    "NIFTY_OPTION": 0.05,        # NIFTY options (CE/PE) trade in 0.05 ticks
    "BANKNIFTY_SPOT": 1.0,       # BankNifty spot/futures
    "BANKNIFTY_OPTION": 0.05,    # BankNifty options
    "FINNIFTY_SPOT": 1.0,        # FinNifty spot/futures
    "FINNIFTY_OPTION": 0.05,     # FinNifty options
}


def get_tick_size(instrument: str, is_option: bool = False) -> float:
    """
    Get correct tick size for instrument and product type.
    
    Args:
        instrument: Instrument name (e.g., "NIFTY", "BANKNIFTY", "FINNIFTY")
        is_option: True if trading options (CE/PE), False for spot/futures
    
    Returns:
        Tick size (1.0 for spot, 0.05 for options)
    
    Example:
        >>> get_tick_size("NIFTY", is_option=False)  # Spot/Futures
        1.0
        >>> get_tick_size("NIFTY", is_option=True)   # Options
        0.05
    """
    base = instrument.upper()
    
    # Normalize instrument name
    if "BANK" in base:
        key = "BANKNIFTY"
    elif "FIN" in base:
        key = "FINNIFTY"
    else:
        key = "NIFTY"
    
    suffix = "_OPTION" if is_option else "_SPOT"
    tick_size = TICK_SIZE_MAP.get(f"{key}{suffix}", 1.0)
    
    return tick_size

# Trading hours (IST)
MARKET_OPEN_TIME = "09:15"  # Market opens
ANALYSIS_START_TIME = os.getenv("ANALYSIS_START_TIME", "09:05")  # Analysis trigger
MARKET_CLOSE_TIME = "15:30"  # Market closes

# Timeframes for analysis
TIMEFRAMES = {
    "5MIN": "5m",
    "15MIN": "15m",
    "1HOUR": "1h",
    "1DAY": "d",
}

PRIMARY_TIMEFRAME = "5MIN"  # Main breakout detection
CONFIRMATION_TIMEFRAME = "15MIN"  # Trend confirmation

# ============================================================================
# TECHNICAL ANALYSIS THRESHOLDS
# ============================================================================

# Volume Configuration
MIN_VOLUME_RATIO = float(os.getenv("MIN_VOLUME_RATIO", 1.5))  # 1.5x avg volume
VOLUME_PERIOD = 20  # candles lookback

# Breakout Configuration
BREAKOUT_CONFIRMATION_CANDLES = 2
FALSE_BREAKOUT_RETRACEMENT = float(os.getenv("MAX_FALSE_BREAKOUT_PERCENT", 0.5))
RETEST_ZONE_PERCENT = float(os.getenv("RETEST_ZONE_PERCENT", 0.15))

# Support/Resistance Configuration
SR_CLUSTER_TOLERANCE = 0.1  # 0.1% difference = same cluster
MIN_SR_TOUCHES = 2
LOOKBACK_BARS = 2000  # Increased to use full 5-day history (was 100)

# Momentum Configuration
MIN_RSI_BULLISH = int(os.getenv("MIN_MOMENTUM_RSI", 50))  # Lowered from 60 to catch early trends
MAX_RSI_BEARISH = int(os.getenv("MAX_MOMENTUM_RSI", 40))
RSI_PERIOD = 14

# ATR (for dynamic SL/TP)
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 2.5

# Moving Averages
EMA_SHORT = 9
EMA_LONG = 21
SMA_VOLUME = 20

# MACD Configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# EMA Crossover Configuration (for combo strategy)
EMA_CROSSOVER_FAST = int(os.getenv("EMA_CROSSOVER_FAST", 5))
EMA_CROSSOVER_SLOW = int(os.getenv("EMA_CROSSOVER_SLOW", 15))

# Combo Signal Feature Flag
USE_COMBO_SIGNALS = os.getenv("USE_COMBO_SIGNALS", "True").lower() == "true"

# Bollinger Bands
BB_PERIOD = 20
BB_STD_DEV = 2

# Stochastic Configuration
STOCH_PERIOD = 14
STOCH_SMOOTH_K = 3
STOCH_SMOOTH_D = 3

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

MIN_RISK_REWARD_RATIO = 1.5
MAX_DAILY_TRADES = 999  # Effectively unlimited (rely on robust filtering)
MIN_SIGNAL_CONFIDENCE = int(os.getenv("MIN_SIGNAL_CONFIDENCE", 65))
MIN_SCORE_THRESHOLD = int(os.getenv("MIN_SCORE_THRESHOLD", 65))  # Raised from 60 for expert-level quality

# Feature Flag: Expert Analysis Enhancements
# Set to False to rollback to previous scoring without code changes
USE_EXPERT_ENHANCEMENTS = os.getenv("USE_EXPERT_ENHANCEMENTS", "True").lower() == "true"

DEFAULT_POSITION_SIZE = 1
MAX_POSITION_SIZE = 3

# ============================================================================
# AI & GROQ CONFIGURATION
# ============================================================================

GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", 400))
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", 0.3))

# Approximate token budget (for your own tracking; not enforced by API)
GROQ_REQUESTS_PER_DAY = 30000
GROQ_REQUESTS_PER_MINUTE = 300

# ============================================================================
# VERTEX AI CONFIGURATION (Gemini)
# ============================================================================

# AI Provider Selection: GROQ, VERTEX, or HYBRID (A/B testing)
AI_PROVIDER = os.getenv("AI_PROVIDER", "GROQ")  # Options: GROQ, VERTEX, HYBRID

# Vertex AI Settings
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "nifty-trading-agent"))
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.0-flash-exp")

# Hybrid Mode Configuration (when AI_PROVIDER=HYBRID)
HYBRID_GROQ_WEIGHT = float(os.getenv("HYBRID_GROQ_WEIGHT", 0.5))    # 50% to Groq
HYBRID_VERTEX_WEIGHT = float(os.getenv("HYBRID_VERTEX_WEIGHT", 0.5))  # 50% to Vertex

# ============================================================================
# MACHINE LEARNING CONFIGURATION
# ============================================================================

# Enable/Disable ML Filtering
USE_ML_FILTERING = os.getenv("USE_ML_FILTERING", "False").lower() == "true"

# Google Cloud Storage for Models
ML_MODEL_BUCKET = os.getenv("ML_MODEL_BUCKET", "nifty-trading-agent-ml-models")
ML_MODEL_NAME = os.getenv("ML_MODEL_NAME", "signal_quality_v1.txt")

# ML Prediction Thresholds
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", 0.65))  # Min probability to accept signal
ML_FALLBACK_TO_RULES = bool(os.getenv("ML_FALLBACK_TO_RULES", "True"))  # Use rule-based if ML unavailable

# Retraining Configuration  
ML_RETRAIN_FREQUENCY_DAYS = int(os.getenv("ML_RETRAIN_FREQUENCY", 7))  # Weekly
ML_MIN_TRAINING_SAMPLES = int(os.getenv("ML_MIN_TRAINING_SAMPLES", 100))  # Minimum trades to train

# ============================================================================
# TELEGRAM CONFIGURATION
# ============================================================================

TELEGRAM_MAX_MESSAGE_LENGTH = 4000
INCLUDE_CHARTS_IN_ALERT = (
    os.getenv("INCLUDE_CHARTS_IN_ALERT", "true").lower() == "true"
)
INCLUDE_AI_SUMMARY_IN_ALERT = (
    os.getenv("INCLUDE_AI_SUMMARY_IN_ALERT", "true").lower() == "true"
)

ALERT_TYPES = {
    "BREAKOUT": "🚀 BREAKOUT",
    "BREAKOUT_CONFIRMED": "✅ BREAKOUT CONFIRMED",
    "FALSE_BREAKOUT": "⚠️ FALSE BREAKOUT",
    "RETEST": "🎯 RETEST SETUP",
    "INSIDE_BAR": "📊 INSIDE BAR SETUP",
    "BREAKDOWN": "📉 BREAKDOWN",
    "SUPPORT_HIT": "🛡️ SUPPORT HIT",
    "RESISTANCE_HIT": "🔴 RESISTANCE HIT",
}

# ============================================================================
# SCHEDULING CONFIGURATION
# ============================================================================

TIME_ZONE = os.getenv("TIME_ZONE", "Asia/Kolkata")
SCHEDULE_CRON = "5 9 * * MON-FRI"
INTRADAY_MONITORING_INTERVAL = 5  # minutes

# ============================================================================
# ============================================================================
# CLOUD DEPLOYMENT
# ============================================================================

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
CLOUD_FUNCTION_REGION = os.getenv("CLOUD_FUNCTION_REGION", "us-central1")
FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE", "")

# Auto-detect GCP Environment (Cloud Run / Cloud Functions)
IS_GCP = os.getenv("K_SERVICE") is not None or os.getenv("FUNCTION_NAME") is not None

# Override DEPLOYMENT_MODE if in GCP
if IS_GCP:
    DEPLOYMENT_MODE = "GCP"
else:
    DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "LOCAL")

ENABLE_CLOUD_LOGGING = DEPLOYMENT_MODE != "LOCAL"

# ============================================================================
# DATA STORAGE & CACHING
# ============================================================================

if DEPLOYMENT_MODE == "GCP" or IS_GCP:
    # Google Cloud Functions only allows writing to /tmp
    CACHE_DIR = "/tmp/cache"
    LOG_DIR = "/tmp/logs"
else:
    CACHE_DIR = "./cache"
    LOG_DIR = "./logs"

CACHE_DATA_TTL = 3600  # seconds

LOG_LEVEL = os.getenv("LOGLEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# DEBUG & DEVELOPMENT
# ============================================================================

DEBUG_MODE = os.getenv("DEBUG_MODE", "False") == "True"

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Daily Alert Limits
MAX_ALERTS_PER_DAY = int(os.getenv("MAX_ALERTS_PER_DAY", "999"))  # Effectively unlimited (rely on other filters)
MAX_ALERTS_PER_TYPE = int(os.getenv("MAX_ALERTS_PER_TYPE", "10"))  # Max per signal type
MAX_ALERTS_PER_INSTRUMENT = int(os.getenv("MAX_ALERTS_PER_INSTRUMENT", "15"))  # Max per instrument

# Choppy Market Detection
MIN_ATR_PERCENT = float(os.getenv("MIN_ATR_PERCENT", "0.06"))  # Min volatility for trading (ATR/price %)
MAX_VWAP_CROSSES = int(os.getenv("MAX_VWAP_CROSSES", "4"))  # Max crosses in 10 bars = choppy

# Correlation Limits
MAX_SAME_DIRECTION_ALERTS = int(os.getenv("MAX_SAME_DIRECTION_ALERTS", "3"))  # Max similar directional trades in 15 mins

SEND_TEST_ALERTS = os.getenv("SEND_TEST_ALERTS", "False").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "False").lower() == "true"
VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"

# ============================================================================
# PHASE 4: MANIPULATION DEFENSE
# ============================================================================

# Flash Crash Protection
MAX_1MIN_MOVE_PCT = float(os.getenv("MAX_1MIN_MOVE_PCT", 0.4))  # 0.4% move in 1 min = Freeze
CIRCUIT_BREAKER_PAUSE_MINS = 15

# Expiry Day Gamma Guard ( Thursdays )
EXPIRY_STOP_TIME = "14:00"  # Stop taking fresh entries after this time on Thursdays
EXPIRY_RISK_ATR_MULTIPLIER = 1.0  # Tighter stops on Expiry

# VIX Volatility Guard
VIX_PANIC_LEVEL = 24.0
VIX_LOW_LEVEL = 10.0


def validate_config():
    """Validate critical configuration"""
    errors = []

    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_KEY_HERE":
        errors.append("❌ GROQ_API_KEY not set in .env")

    if (
        not TELEGRAM_BOT_TOKEN
        or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE"
    ):
        errors.append("❌ TELEGRAM_BOT_TOKEN not set in .env")

    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
        errors.append("❌ TELEGRAM_CHAT_ID not set in .env")

    return errors


os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if __name__ == "__main__":
    errs = validate_config()
    if errs:
        print("\n⚠️  Configuration Issues:\n")
        for e in errs:
            print(f"   {e}")
        print("\n📝 Please create .env file with required keys")
    else:
        print("✅ Configuration validated successfully!")

```

## Data Module

### ./data_module/__init__.py

```python

```

### ./data_module/fetcher.py

```python
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
    FYERS_SECRET_ID,
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

        # Initialize Fyers App with OAuth support
        self.fyers_app = FyersApp(app_id=FYERS_CLIENT_ID, secret_id=FYERS_SECRET_ID)
        
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

    # =====================================================================
    # INDIA VIX FETCHING (Phase 3)
    # =====================================================================

    def fetch_india_vix(self) -> Optional[float]:
        """
        Fetch India VIX value for adaptive threshold calculations.
        
        Priority:
        1. yfinance (^NSEIV)
        2. Fyers API (NSE:INDIAVIX-INDEX)
        3. Return None (fall back to ATR percentile)
        
        Returns:
            float: VIX value or None
        """
        # Try cache first (5-minute TTL)
        cache_key = "india_vix"
        if self._is_cache_valid(cache_key, ttl=300):
            return self.cache[cache_key]
        
        # 1. Try yfinance
        vix = self._fetch_vix_yfinance()
        if vix:
            self.cache[cache_key] = vix
            self.cache_time[cache_key] = time.time()
            return vix
        
        # 2. Try Fyers
        vix = self._fetch_vix_fyers()
        if vix:
            self.cache[cache_key] = vix
            self.cache_time[cache_key] = time.time()
            return vix
        
        logger.warning("⚠️ VIX fetch failed from all sources, will use ATR percentile fallback")
        return None

    def _fetch_vix_yfinance(self) -> Optional[float]:
        """Fetch VIX from yfinance."""
        try:
            vix_ticker = yf.Ticker("^NSEIV")
            info = vix_ticker.info
            vix_value = info.get('regularMarketPrice') or info.get('previousClose')
            
            if vix_value and vix_value > 0:
                logger.debug(f"📊 VIX (yfinance): {vix_value:.2f}")
                return float(vix_value)
            return None
        except Exception as e:
            logger.debug(f"VIX yfinance fetch failed: {e}")
            return None

    def _fetch_vix_fyers(self) -> Optional[float]:
        """Fetch VIX from Fyers API."""
        try:
            # Fyers symbol for India VIX
            quote = self.fyers_app.get_quote("NSE:INDIAVIX-INDEX")
            if not quote:
                return None
            
            data_source = quote.get('v', quote) if isinstance(quote.get('v'), dict) else quote
            vix_value = data_source.get('lp') or data_source.get('last_price')
            
            if vix_value and vix_value > 0:
                logger.debug(f"📊 VIX (Fyers): {vix_value:.2f}")
                return float(vix_value)
            return None
        except Exception as e:
            logger.debug(f"VIX Fyers fetch failed: {e}")
            return None

    def _fetch_fyers_data(self, instrument: str) -> Optional[Dict]:
        """Fetch from Fyers."""
        try:
            quote = self.fyers_app.get_quote(instrument)
            if not quote:
                self._send_fyers_failure_notification()
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
            self._send_fyers_failure_notification()
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

            # Flatten multi-level columns if present (yfinance returns ('Close', '^NSEI') format)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Convert column names to lowercase for consistent access
            df.columns = df.columns.str.lower()

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
    
    def _send_fyers_failure_notification(self):
        """Send Telegram notification when Fyers fails (once per day)."""
        try:
            from data_module.persistence import get_persistence
            
            persistence = get_persistence()
            stats = persistence.get_daily_stats()
            
            # Check if we've already sent the alert today
            if stats.get("fyers_failure_alert_sent"):
                return
            
            # Import here to avoid circular dependency
            from telegram_module.bot_handler import get_bot
            
            message = (
                "⚠️ <b>Fyers API Failure Alert</b>\n\n"
                "Fyers API authentication has failed. The system is automatically using yfinance as a fallback.\n\n"
                "<b>Action Required:</b>\n"
                "1. Go to https://myapi.fyers.in/dashboard\n"
                "2. Generate a new Access Token\n"
                "3. Update Secret Manager:\n\n"
                "<code>echo -n 'YOUR_NEW_TOKEN' | gcloud secrets versions add fyers-access-token --data-file=-</code>\n\n"
                "📊 <i>Note: Trading continues normally with yfinance data</i>"
            )
            
            bot = get_bot()
            bot.send_message(message)
            
            # Mark as sent for today
            persistence.increment_stat("fyers_failure_alert_sent")
            logger.info("📨 Sent Fyers failure Telegram notification")
            
        except Exception as e:
            logger.debug(f"Failed to send Fyers notification: {e}")

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

```

### ./data_module/fyers_interface.py

```python

import logging
import os
from typing import Dict, Optional, Any
from fyers_apiv3 import fyersModel
import time

logger = logging.getLogger(__name__)

# Try to import OAuth manager (optional, graceful fallback)
try:
    from data_module.fyers_oauth import get_oauth_manager
    OAUTH_AVAILABLE = True
except ImportError:
    OAUTH_AVAILABLE = False
    logger.debug("OAuth manager not available, using basic token auth")

class FyersApp:
    def __init__(self, app_id: str, secret_id: Optional[str] = None, access_token: Optional[str] = None):
        self.client_id = app_id  # App ID (e.g., DURQKS8D17-100)
        self.secret_key = secret_id
        self.access_token = access_token
        self.fyers: Optional[fyersModel.FyersModel] = None
        self.oauth_manager = None
        self.mapper = {
            "NIFTY": "NSE:NIFTY50-INDEX",
            "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
            "FINNIFTY": "NSE:FINNIFTY-INDEX",
            "NSE:INDIAVIX-INDEX": "NSE:INDIAVIX-INDEX"
        }
        
        # Try to use OAuth if available
        if OAUTH_AVAILABLE and secret_id:
            try:
                self.oauth_manager = get_oauth_manager()
                logger.info("🔐 Using OAuth manager for automatic token refresh")
            except Exception as e:
                logger.debug(f"OAuth manager not initialized: {e}")
        
        self.initialize_session()

    def initialize_session(self):
        """
        Initialize Fyers Model.
        Requires access_token. If not provided, it attempts to read from env or OAuth.
        """
        # Try OAuth first if available
        if self.oauth_manager and self.oauth_manager.is_authorized():
            self.access_token = self.oauth_manager.get_valid_access_token()
            if self.access_token:
                logger.info("✅ Using OAuth-managed access token")
        
        # Fallback to environment variable
        if not self.access_token:
            # Try to load from env
            self.access_token = os.getenv("FYERS_ACCESS_TOKEN")
        
        if not self.access_token:
            # If no token, we can't make authenticated calls
            logger.warning("⚠️ Fyers Access Token is missing. Fyers API calls will fail.")
            return

        try:
            self.fyers = fyersModel.FyersModel(
                client_id=self.client_id,
                token=self.access_token,
                is_async=False, 
                log_path=os.getcwd()
            )
            logger.info("✅ Fyers Model initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Fyers Model: {e}")

    def validate_session(self) -> bool:
        """
        Validate if the current Fyers session is active.
        Returns True if session is valid, False otherwise.
        """
        if not self.fyers:
            return False
        
        try:
            # Try a simple API call to check token validity
            test_data = {"symbols": "NSE:NIFTY50-INDEX"}
            response = self.fyers.quotes(data=test_data)
            
            if response.get("s") == "ok":
                logger.debug("✅ Fyers session is valid")
                return True
            else:
                error_msg = response.get('message', 'Unknown')
                if 'token' in error_msg.lower() or 'invalid' in error_msg.lower():
                    logger.warning(f"⚠️ Fyers token expired or invalid: {error_msg}")
                    return False
                return False
        except Exception as e:
            logger.warning(f"⚠️ Fyers session validation failed: {e}")
            return False
    
    def refresh_token_from_env(self) -> bool:
        """
        Attempt to refresh token from environment variable.
        Useful if token is updated externally.
        Returns True if new token loaded successfully.
        """
        new_token = os.getenv("FYERS_ACCESS_TOKEN")
        if new_token and new_token != self.access_token:
            logger.info("🔄 Attempting to refresh Fyers token from environment")
            self.access_token = new_token
            self.initialize_session()
            return self.validate_session()
        return False

    def get_option_chain(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch option chain for the given symbol.
        Args:
            symbol: Common symbol name (e.g., "NIFTY", "BANKNIFTY")
        """
        if not self.fyers:
            logger.error("❌ Fyers session not initialized")
            return None
            
        fyers_symbol = self.mapper.get(symbol)
        if not fyers_symbol:
            logger.error(f"❌ Unknown symbol for Fyers: {symbol}")
            return None
            
        try:
            # According to Fyers API v3 docs
            data = {
                "symbol": fyers_symbol,
                "strikecount": 20, # Fetch adequate strikes
                "timestamp": ""
            }
            
            response = self.fyers.optionchain(data=data)
            
            if response.get("s") == "ok":
                logger.info(f"✅ Fyers Option Chain fetched for {symbol}")
                return response
            else:
                logger.error(f"❌ Fyers API Error: {response.get('message', 'Unknown Error')}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Exception fetching Fyers Option Chain: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch Level 1 quote (LTP, OHLC) for a symbol.
        """
        if not self.fyers:
            # Try re-initializing if token is available
            self.initialize_session()
            if not self.fyers:
                logger.debug("Fyers not available, caller will use fallback")
                return None
        
        # Validate session before making call
        if not self.validate_session():
            logger.warning("⚠️ Fyers session invalid, trying token refresh...")
            if not self.refresh_token_from_env():
                logger.info("ℹ️ Fyers unavailable - system will use yfinance fallback")
                return None
                
        fyers_symbol = self.mapper.get(symbol)
        if not fyers_symbol:
            logger.error(f"❌ Unknown symbol: {symbol}")
            return None

        try:
            data = {"symbols": fyers_symbol}
            response = self.fyers.quotes(data=data)
            
            if response.get("s") == "ok" and "d" in response:
                return response["d"][0] # Return the first (and only) result
            else:
                logger.error(f"❌ Fyers Quote Failed: {response.get('message')}")
                return None
        except Exception as e:
            logger.error(f"❌ Exception fetching quote: {e}")
            return None

if __name__ == "__main__":
    # Test Block
    APP_ID = "DURQKS8D17-100"
    SECRET_ID = os.getenv("FYERS_SECRET_ID") # User said they have it
    ACCESS_TOKEN = os.getenv("FYERS_ACCESS_TOKEN")
    
    app = FyersApp(app_id=APP_ID, secret_id=SECRET_ID, access_token=ACCESS_TOKEN)
    if app.fyers:
        print("Testing Fetch...")
        data = app.get_option_chain("NIFTY")
        print(data)

```

### ./data_module/fyers_oauth.py

```python
# Fyers OAuth Token Manager
# Handles automatic token refresh using OAuth refresh tokens

import logging
import os
import json
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
import requests

logger = logging.getLogger(__name__)


class FyersOAuthManager:
    """
    Manages Fyers OAuth tokens with automatic refresh capability.
    Stores refresh token in Cloud Secret Manager for persistence.
    """
    
    def __init__(self, app_id: str, secret_key: str, redirect_uri: str = "https://trade.fyers.in/api-login/redirect-uri/index.html"):
        self.app_id = app_id
        self.secret_key = secret_key
        self.redirect_uri = redirect_uri
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expiry: Optional[datetime] = None
        
        # Try to load existing tokens
        self._load_tokens()
    
    def _load_tokens(self):
        """Load tokens from environment or Cloud Secret Manager."""
        # Try environment first (for local development)
        self.access_token = os.getenv("FYERS_ACCESS_TOKEN")
        self.refresh_token = os.getenv("FYERS_REFRESH_TOKEN")
        self.pin = os.getenv("FYERS_PIN", "")
        
        if self.access_token:
            logger.info(f"📝 Loaded access token from environment (length: {len(self.access_token)})")
        
        if self.pin:
            logger.info("🔑 Fyers PIN found in environment")
        
        if self.refresh_token:
            logger.info(f"📝 Loaded refresh token from environment (length: {len(self.refresh_token)})")
        else:
            logger.info("⚠️ No refresh token in environment, trying Secret Manager...")
            # Try Cloud Secret Manager if in production
            self.refresh_token = self._load_from_secret_manager("fyers-refresh-token")
            
        if self.refresh_token:
            logger.info("✅ Refresh token available for automatic token refresh")
        else:
            logger.warning("⚠️ No refresh token found - automatic refresh not available")
    
    def _load_from_secret_manager(self, secret_name: str) -> Optional[str]:
        """Load secret from Google Cloud Secret Manager."""
        try:
            from google.cloud import secretmanager
            
            project_id = os.getenv("GCP_PROJECT_ID", "nifty-trading-agent")
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            
            response = client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            
            logger.info(f"✅ Loaded {secret_name} from Secret Manager")
            return secret_value
        except Exception as e:
            logger.debug(f"Could not load {secret_name} from Secret Manager: {e}")
            return None
    
    def _save_to_secret_manager(self, secret_name: str, secret_value: str) -> bool:
        """Save secret to Google Cloud Secret Manager."""
        try:
            from google.cloud import secretmanager
            
            project_id = os.getenv("GCP_PROJECT_ID", "nifty-trading-agent")
            client = secretmanager.SecretManagerServiceClient()
            parent = f"projects/{project_id}"
            secret_id = secret_name
            
            # Check if secret exists
            try:
                secret_path = f"{parent}/secrets/{secret_id}"
                client.get_secret(request={"name": secret_path})
                exists = True
            except:
                exists = False
            
            # Create secret if it doesn't exist
            if not exists:
                client.create_secret(
                    request={
                        "parent": parent,
                        "secret_id": secret_id,
                        "secret": {"replication": {"automatic": {}}},
                    }
                )
            
            # Add new version
            parent_path = f"{parent}/secrets/{secret_id}"
            payload = secret_value.encode("UTF-8")
            client.add_secret_version(
                request={"parent": parent_path, "payload": {"data": payload}}
            )
            
            logger.info(f"✅ Saved {secret_name} to Secret Manager")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save {secret_name} to Secret Manager: {e}")
            return False
    
    def generate_auth_url(self) -> str:
        """
        Generate Fyers authorization URL for initial OAuth flow.
        User needs to visit this URL and authorize the app.
        """
        session_model = fyersModel.SessionModel(
            client_id=self.app_id,
            secret_key=self.secret_key,
            redirect_uri=self.redirect_uri,
            response_type="code",
            grant_type="authorization_code"
        )
        
        auth_url = session_model.generate_authcode()
        logger.info(f"🔐 Authorization URL generated")
        return auth_url
    
    def exchange_auth_code(self, auth_code: str) -> Tuple[bool, str]:
        """
        Exchange authorization code for access token and refresh token.
        
        Args:
            auth_code: The authorization code from Fyers OAuth callback
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            session_model = fyersModel.SessionModel(
                client_id=self.app_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type="code",
                grant_type="authorization_code"
            )
            
            session_model.set_token(auth_code)
            response = session_model.generate_token()
            
            if response.get("code") == 200:
                self.access_token = response.get("access_token")
                self.refresh_token = response.get("refresh_token")
                
                # Calculate expiry (Fyers tokens typically last 24 hours)
                self.token_expiry = datetime.now() + timedelta(hours=23, minutes=50)
                
                # Save refresh token to Secret Manager for persistence
                self._save_to_secret_manager("fyers-refresh-token", self.refresh_token)
                
                logger.info("✅ Successfully exchanged auth code for tokens")
                return True, "Tokens obtained successfully"
            else:
                error_msg = response.get("message", "Unknown error")
                logger.error(f"❌ Token exchange failed: {error_msg}")
                return False, error_msg
                
        except Exception as e:
            logger.error(f"❌ Exception during token exchange: {e}")
            return False, str(e)
    
    def refresh_access_token(self) -> Tuple[bool, str]:
        """
        Refresh the access token using the refresh token.
        Uses Fyers V3 API endpoint: https://api-t1.fyers.in/api/v3/validate-refresh-token
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.refresh_token:
            logger.error("❌ No refresh token available")
            return False, "No refresh token available"
        
        try:
            import hashlib
            
            # Fyers V3 requires appIdHash (SHA-256 of app_id:secret_key)
            app_id_hash = hashlib.sha256(f"{self.app_id}:{self.secret_key}".encode()).hexdigest()
            
            # Fyers V3 refresh endpoint
            url = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
            
            payload = {
                "grant_type": "refresh_token",
                "appIdHash": app_id_hash,
                "refresh_token": self.refresh_token,
                "pin": self.pin
            }
            
            logger.info("🔄 Attempting to refresh Fyers access token...")
            response = requests.post(url, json=payload)
            response_data = response.json()
            
            if response_data.get("code") == 200:
                self.access_token = response_data.get("access_token")
                
                # Fyers doesn't return new refresh token on refresh, same one is reused
                # Refresh token expires after 15 days
                
                # Update expiry (tokens valid for 24 hours)
                self.token_expiry = datetime.now() + timedelta(hours=23, minutes=50)
                
                logger.info("✅ Access token refreshed successfully via Fyers API")
                return True, "Token refreshed successfully"
            else:
                error_msg = response_data.get("message", "Unknown error")
                logger.error(f"❌ Token refresh failed: {error_msg}")
                logger.debug(f"Response: {response_data}")
                return False, error_msg
                
        except Exception as e:
            logger.error(f"❌ Exception during token refresh: {e}")
            return False, str(e)
    
    def get_valid_access_token(self) -> Optional[str]:
        """
        Get a valid access token, automatically refreshing if needed.
        This is the main method to call when you need an access token.
        
        Returns:
            Valid access token or None if refresh failed
        """
        # If no access token, try to refresh immediately
        if not self.access_token:
            logger.info("⚠️ No access token available, attempting refresh...")
            if self.refresh_token:
                success, message = self.refresh_access_token()
                if success:
                    return self.access_token
                else:
                    logger.error(f"❌ Token refresh failed: {message}")
                    return None
            else:
                logger.error("❌ No refresh token available - need to re-authorize")
                return None
        
        # Conservative: within 1 hour
        # TRUST THE TOKEN: If we have an access_token but token_expiry is None,
        # it means we just loaded it from .env. Trust it for the first hour 
        # instead of immediately trying to refresh it (which might fail).
        if self.token_expiry is None:
            # Set a default expiry 12 hours from now for tokens loaded from env
            self.token_expiry = datetime.now() + timedelta(hours=12)
            logger.info("💎 Trusting existing access token for 12 hours")
            return self.access_token

        if datetime.now() >= self.token_expiry - timedelta(hours=1):
            logger.info("🔄 Access token may be expired, attempting refresh...")
            if self.refresh_token:
                success, message = self.refresh_access_token()
                if not success:
                    logger.warning(f"⚠️ Token refresh failed: {message}, trying existing token")
            else:
                logger.warning("⚠️ No refresh token for auto-refresh")
        
        return self.access_token
    
    def is_authorized(self) -> bool:
        """Check if we have valid authorization (refresh token available)."""
        return self.refresh_token is not None


# Singleton instance
_oauth_manager: Optional[FyersOAuthManager] = None


def get_oauth_manager() -> FyersOAuthManager:
    """Get or create the OAuth manager singleton."""
    global _oauth_manager
    
    if _oauth_manager is None:
        app_id = os.getenv("FYERS_CLIENT_ID", "")
        secret_key = os.getenv("FYERS_SECRET_ID", "")
        
        if not app_id or not secret_key:
            raise ValueError("FYERS_CLIENT_ID and FYERS_SECRET_ID must be set")
        
        _oauth_manager = FyersOAuthManager(app_id, secret_key)
    
    return _oauth_manager

```

### ./data_module/ml_data_collector.py

```python
"""
ML Data Collector
Collects and stores training data in Firestore
"""

import logging
from typing import Dict, Optional
from datetime import datetime
import pytz

from ml_module.feature_extractor import extract_features

logger = logging.getLogger(__name__)


class MLDataCollector:
    """Collect ML training data from live trading signals."""
    
    def __init__(self, firestore_client):
        """
        Initialize data collector.
        
        Args:
            firestore_client: Firestore client instance
        """
        self.db = firestore_client
        self.collection_name = "ml_training_data"
        
        logger.info("✅ ML Data Collector initialized")
    
    def record_signal(
        self,
        signal: Dict,
        technical_context: Dict,
        option_metrics: Dict,
        market_status: Dict = None
    ) -> Optional[str]:
        """
        Record a signal for ML training.
        
        Args:
            signal: Trading signal
            technical_context: MTF analysis context
            option_metrics: Options data
            market_status: Market conditions
            
        Returns:
            Document ID if successful
        """
        try:
            # Extract features
            features = extract_features(
                signal,
                technical_context,
                option_metrics,
                market_status
            )
            
            # Create training record
            ist = pytz.timezone("Asia/Kolkata")
            
            record = {
                "signal_id": signal.get("timestamp", datetime.now(ist).isoformat()),
                "instrument": signal.get("instrument"),
                "signal_type": signal.get("signal_type"),
                "features": features,
                "label": None,  # Will be updated when outcome is known
                "outcome": None,  # WIN/LOSS/PENDING
                "entry_price": signal.get("entry_price"),
                "stop_loss": signal.get("stop_loss"),
                "take_profit": signal.get("take_profit"),
                "timestamp": datetime.now(ist),
                "outcome_timestamp": None,
                "metadata": {
                    "confidence": signal.get("confidence"),
                    "score": signal.get("score"),
                    "ml_probability": signal.get("ml_probability")  # If ML was used
                }
            }
            
            # Store in Firestore
            doc_ref = self.db.collection(self.collection_name).document()
            doc_ref.set(record)
            
            logger.info(f"📝 ML training data recorded: {doc_ref.id}")
            return doc_ref.id
            
        except Exception as e:
            logger.error(f"❌ Failed to record ML training data: {e}")
            return None
    
    def update_outcome(
        self,
        signal_id: str,
        outcome: str,
        actual_exit_price: float = None
    ) -> bool:
        """
        Update signal outcome when trade closes.
        
        Args:
            signal_id: Document ID or signal timestamp
            outcome: "WIN" or "LOSS"
            actual_exit_price: Exit price if available
            
        Returns:
            True if successful
        """
        try:
            # Find document by signal_id
            docs = self.db.collection(self.collection_name).where(
                "signal_id", "==", signal_id
            ).limit(1).get()
            
            if not docs:
                logger.warning(f"Signal not found: {signal_id}")
                return False
            
            doc = docs[0]
            
            # Update with outcome
            ist = pytz.timezone("Asia/Kolkata")
            doc.reference.update({
                "outcome": outcome,
                "label": 1 if outcome == "WIN" else 0,
                "outcome_timestamp": datetime.now(ist),
                "actual_exit_price": actual_exit_price
            })
            
            logger.info(f"✅ Updated outcome for {signal_id}: {outcome}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update outcome: {e}")
            return False
    
    def get_training_data(
        self,
        days: int = 90,
        min_samples: int = 50
    ) -> list:
        """
        Fetch completed training samples from Firestore.
        
        Args:
            days: Number of days to look back
            min_samples: Minimum samples required
            
        Returns:
            List of training records with labels
        """
        try:
            from datetime import timedelta
            ist = pytz.timezone("Asia/Kolkata")
            cutoff = datetime.now(ist) - timedelta(days=days)
            
            # Query completed records (outcome != None)
            docs = self.db.collection(self.collection_name).where(
                "outcome_timestamp", ">=", cutoff
            ).where(
                "label", "!=", None
            ).get()
            
            training_data = []
            for doc in docs:
                data = doc.to_dict()
                if data.get("label") is not None:  # Ensure labeled
                    training_data.append(data)
            
            logger.info(f"📊 Fetched {len(training_data)} training samples")
            
            if len(training_data) < min_samples:
                logger.warning(
                    f"⚠️ Only {len(training_data)} samples available "
                    f"(min: {min_samples})"
                )
            
            return training_data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch training data: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get statistics on collected data."""
        try:
            # Total records
            total = len(self.db.collection(self.collection_name).get())
            
            # Labeled records
            labeled = len(
                self.db.collection(self.collection_name)
                .where("label", "!=", None)
                .get()
            )
            
            # Pending labels
            pending = total - labeled
            
            return {
                "total_records": total,
                "labeled_records": labeled,
                "pending_labels": pending,
                "collection": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {}

```

### ./data_module/option_chain_fetcher.py

```python

import requests
import time
import logging
from typing import Dict, Optional
from datetime import datetime
from config.settings import FYERS_CLIENT_ID
from data_module.fyers_interface import FyersApp

logger = logging.getLogger(__name__)

class OptionChainFetcher:
    """
    Fetches option chain data from NSE official API.
    Handles session management and headers to mimic a browser.
    """
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/option-chain'
        })
        self.cache = {}
        self.cache_time = {}
        self.cache_ttl = 300  # 5 minutes (increased from 60 for production stability)
        
        # NEW: Emergency cache for fallback when both Fyers AND NSE fail
        self.last_valid_data = {}
        self.degraded_mode = False
        
        # Initial visit to set cookies
        try:
            self.session.get("https://www.nseindia.com", timeout=10)
        except Exception as e:
            logger.warning(f"⚠️ Initial NSE visit failed: {e}")
            
        # Initialize Fyers App for PRIMARY data source
        self.fyers_app = FyersApp(app_id=FYERS_CLIENT_ID)

    def _is_cache_valid(self, key: str) -> bool:
        if key in self.cache and key in self.cache_time:
            if time.time() - self.cache_time[key] < self.cache_ttl:
                return True
        return False

    def fetch_option_chain(self, instrument: str) -> Optional[Dict]:
        """
        Fetch option chain for NIFTY or BANKNIFTY.
        Prioritizes Fyers API. Falls back to NSE website if Fyers fails.
        
        Args:
            instrument: Symbol name (e.g., "NIFTY", "BANKNIFTY")
            
        Returns:
            Dictionary containing option chain data or None if failed.
        """
        # Map instrument to NSE symbol format
        symbol = "NIFTY" if "NIFTY" in instrument and "BANK" not in instrument else "BANKNIFTY"
        if "FIN" in instrument: symbol = "FINNIFTY"
        
        cache_key = f"oc_{symbol}"
        if self._is_cache_valid(cache_key):
            self.degraded_mode = False  # Reset if cache is working
            return self.cache[cache_key]

        # ----------------------------------------
        # PRIMARY: FYERS API
        # ----------------------------------------
        data = self.fetch_fyers_data(instrument)
        if data:
            self.cache[cache_key] = data
            self.cache_time[cache_key] = time.time()
            # Store as emergency backup
            self.last_valid_data[cache_key] = data
            self.degraded_mode = False
            return data

        # ----------------------------------------
        # FALLBACK: NSE WEBSITE (Scraping)
        # ----------------------------------------
        logger.warning(f"⚠️ Fyers Option Chain failed. Falling back to NSE Scraping for {symbol}")
        
        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        
        try:
            # Add specific headers for API call
            headers = {
                'Referer': f'https://www.nseindia.com/option-chain?symbol={symbol}'
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 401:
                # Refresh session if unauthorized
                logger.info("🔄 Refreshing NSE session...")
                self.session.get("https://www.nseindia.com", timeout=10)
                response = self.session.get(url, headers=headers, timeout=10)

            response.raise_for_status()
            data = response.json()
            
            # Validate Data
            if "records" not in data or "data" not in data.get("records", {}):
                raise ValueError("Invalid NSE data structure (missing records)")

            # Cache result
            self.cache[cache_key] = data
            self.cache_time[cache_key] = time.time()
            # Store as emergency backup
            self.last_valid_data[cache_key] = data
            self.degraded_mode = False
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch option chain for {symbol}: {e}")
            
            # ----------------------------------------
            # EMERGENCY FALLBACK: Use last known good data
            # ----------------------------------------
            if cache_key in self.last_valid_data:
                logger.warning(f"⚠️ Using STALE option chain data (>5min old) for {symbol}")
                self.degraded_mode = True
                return self.last_valid_data[cache_key]
            
            logger.critical(f"❌ NO OPTION CHAIN DATA AVAILABLE for {symbol}")
            self.degraded_mode = True
            return None
    
    def is_healthy(self) -> bool:
        """Check if option chain fetcher is working normally."""
        return not self.degraded_mode

    def fetch_fyers_data(self, instrument: str) -> Optional[Dict]:
        """Fetch and transform data from Fyers."""
        try:
            raw_data = self.fyers_app.get_option_chain(instrument)
            if raw_data and raw_data.get('data'):
                return self._transform_fyers_to_nse(raw_data['data'])
            return None
        except Exception as e:
            logger.error(f"❌ Fyers Fallback failed: {e}")
            return None

    def _transform_fyers_to_nse(self, fyers_data: Dict) -> Dict:
        """
        Transform Fyers response to match NSE structure for compatibility.
        """
        options_chain = fyers_data.get('optionsChain', [])
        expiry_data = fyers_data.get('expiryData', [])
        
        grouped = {}
        for item in options_chain:
            strike = item.get('strike_price')
            if not strike or strike <= 0: continue
            
            if strike not in grouped:
                grouped[strike] = {'strikePrice': strike}
            
            # Determine type
            sym = item.get('symbol', '')
            if 'CE' in sym[-2:]: type_key = 'CE'
            elif 'PE' in sym[-2:]: type_key = 'PE'
            else: continue
            
            # Map fields
            # Note: Fyers keys need to be verified. Assuming standard keys here.
            # Adjust keys based on actual Fyers API response inspection if needed.
            node = {
                'strikePrice': strike,
                'openInterest': item.get('oi', 0),
                'changeinOpenInterest': item.get('oich', 0),
                'totalTradedVolume': item.get('volume', 0),
                'impliedVolatility': item.get('iv', 0),
                'lastPrice': item.get('ltp', 0),
                'change': item.get('ltpch', 0), 
                'pChange': item.get('ltpchp', 0)
            }
            grouped[strike][type_key] = node

        unique_expiries = [x.get('date') for x in expiry_data if x.get('date')]
        
        data_list = sorted(list(grouped.values()), key=lambda x: x['strikePrice'])
        
        return {
            'records': {
                'expiryDates': unique_expiries,
                'data': data_list,
                'timestamp':  datetime.now().strftime("%d-%b-%Y %H:%M:%S")
            },
            'filtered': {
                'data': data_list, # Providing full list as filtered for now
                'CE': {'totOI': 0, 'totVol': 0}, 
                'PE': {'totOI': 0, 'totVol': 0}
            }
        }

if __name__ == "__main__":
    # Test execution
    logging.basicConfig(level=logging.INFO)
    fetcher = OptionChainFetcher()
    data = fetcher.fetch_option_chain("NIFTY")
    if data:
        print("✅ Fetch success")
        records = data.get('records', {})
        print(f"Expiry Dates: {records.get('expiryDates', [])[:3]}")
    else:
        print("❌ Fetch failed")

```

### ./data_module/persistence.py

```python
"""
Persistence Module
Handles state storage using Google Cloud Firestore.
Required for stateless Cloud Function execution to track daily stats.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
import json
import pytz

from google.cloud import firestore
from config.settings import TIME_ZONE, DEBUG_MODE

logger = logging.getLogger(__name__)

class PersistenceManager:
    """Manages daily state in Firestore."""
    
    def __init__(self):
        self.db = None
        self.collection_name = "daily_stats"
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        
        if not self.project_id:
            logger.warning("⚠️ GOOGLE_CLOUD_PROJECT not set, persistence disabled")
            return

        try:
            self.db = firestore.Client(project=self.project_id)
            logger.info(f"💾 Firestore initialized | Project: {self.project_id}")
        except Exception as e:
            logger.error(f"❌ Firestore init failed: {str(e)}")

    def _get_today_doc_id(self) -> str:
        """Get document ID for today (YYYY-MM-DD)."""
        return datetime.now().strftime("%Y-%m-%d")

    def increment_stat(self, stat_name: str, value: int = 1):
        """Increment a counter stat."""
        if not self.db:
            return

        try:
            doc_ref = self.db.collection(self.collection_name).document(self._get_today_doc_id())
            
            # Use atomic increment
            doc_ref.set({
                stat_name: firestore.Increment(value),
                "last_updated": firestore.SERVER_TIMESTAMP
            }, merge=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to increment {stat_name}: {str(e)}")

    def _sanitize_for_firestore(self, data):
        """Recursively convert numpy types to standard Python types."""
        try:
            import numpy as np
            
            if isinstance(data, dict):
                return {k: self._sanitize_for_firestore(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [self._sanitize_for_firestore(v) for v in data]
            elif isinstance(data, (np.integer, np.int64, np.int32)):
                return int(data)
            elif isinstance(data, (np.floating, np.float64, np.float32)):
                return float(data)
            elif isinstance(data, np.bool_):
                return bool(data)
            elif hasattr(data, "item"):  # Generic numpy scalar check
                return data.item()
            return data
        except ImportError:
            return data
        except Exception as e:
            logger.warning(f"⚠️ Sanitization failed: {e}")
            return data

    def add_event(self, event_type: str, event_data: Dict):
        """Add an event (breakout, signal) to the daily list."""
        if not self.db:
            return

        try:
            # Sanitize data (convert numpy types to python types)
            clean_data = self._sanitize_for_firestore(event_data)
            
            doc_ref = self.db.collection(self.collection_name).document(self._get_today_doc_id())
            
            # Firestore array_union to append to list
            doc_ref.set({
                f"events.{event_type}": firestore.ArrayUnion([clean_data]),
                "last_updated": firestore.SERVER_TIMESTAMP
            }, merge=True)
            
        except Exception as e:
            logger.error(f"❌ Failed to add event {event_type}: {str(e)}")

    def get_daily_stats(self) -> Dict:
        """Retrieve all stats for today."""
        if not self.db:
            return {}

        try:
            doc_ref = self.db.collection(self.collection_name).document(self._get_today_doc_id())
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            
            # If doc doesn't exist, create it initialized
            initial_stats = {"created_at": firestore.SERVER_TIMESTAMP}
            doc_ref.set(initial_stats, merge=True)
            return initial_stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get daily stats: {str(e)}")
            return {}
    
    
    def on_new_trading_day(self):
        """
        Reset daily stats and alerts if the date has changed.
        Should be called at startup.
        """
        if not self.db: return
        
        try:
            today_id = self._get_today_doc_id()
             # We can't easily "check" if it's a new day without storing "last_run_date"
             # But if we rely on _get_today_doc_id() for daily_stats, we just need to ensure 
             # recent_alerts are cleared if they belong to previous days.
            pass
        except Exception as e:
            logger.error(f"❌ Failed to handle new trading day: {e}")

    def save_recent_alerts(self, recent_alerts: Dict):
        """
        Save recent alerts to Firestore for duplicate detection across executions.
        
        Args:
            recent_alerts: Dict of {AlertKey: timestamp}
        """
        if not self.db:
            return
        
        try:
            doc_ref = self.db.collection("system_state").document("recent_alerts")
            
            # Convert AlertKey objects to string representation for JSON compatibility
            # Keys must be strings in Firestore maps
            alerts_serialized = {}
            for key, timestamp in recent_alerts.items():
                k_str = str(key) if not isinstance(key, str) else key
                alerts_serialized[k_str] = timestamp.isoformat()
            
            doc_ref.set({
                "alerts": alerts_serialized,
                "updated_at": firestore.SERVER_TIMESTAMP
            })
            
            logger.debug(f"💾 Saved {len(recent_alerts)} recent alerts to Firestore")
        
        except Exception as e:
            logger.error(f"❌ Failed to save recent alerts: {e}")
    
    def get_recent_alerts(self) -> Dict:
        """
        Retrieve recent alerts from Firestore.
        
        Returns:
            Dict of {AlertKey: timestamp}
        """
        from data_module.persistence_models import AlertKey
        
        if not self.db:
            return {}
        
        recent_alerts = {}
        try:
            doc_ref = self.db.collection("system_state").document("recent_alerts")
            doc = doc_ref.get()
            
            if not doc.exists:
                return {}
            
            data = doc.to_dict()
            alerts_serialized = data.get("alerts", {})
            
            ist = pytz.timezone("Asia/Kolkata")
            today = datetime.now(ist).strftime("%Y-%m-%d")
            
            for key_str, timestamp_str in alerts_serialized.items():
                try:
                    # Parse timestamp
                    dt = datetime.fromisoformat(timestamp_str)
                    if dt.tzinfo is None:
                        dt = ist.localize(dt)
                    
                    # Parse Key string back to AlertKey if possible
                    # Format: instrument|signal_type|level_ticks|date
                    parts = key_str.split("|")
                    if len(parts) == 4:
                        if parts[3] != today:
                            # Skip alerts from previous days
                            continue
                            
                        alert_key = AlertKey(
                            instrument=parts[0],
                            signal_type=parts[1],
                            level_ticks=int(parts[2]),
                            date=parts[3]
                        )
                        recent_alerts[alert_key] = dt
                    else:
                        # Legacy string key support (or if format changes)
                        # We might choose to drop legacy keys to force clean state
                        pass 
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse alert entry {key_str}: {e}")
            
            logger.info(f"📂 Loaded {len(recent_alerts)} active alerts for today from Firestore")
            return recent_alerts
        
        except Exception as e:
            logger.error(f"❌ Failed to get recent alerts: {e}")
            # Fail Closed: Return empty dict means we might re-alert if DB is down but memory is empty.
            # However, main.py will populate memory as it runs.
            return {}

# =========================================================================
# SINGLETON
# =========================================================================

_persistence: Optional[PersistenceManager] = None


def get_persistence() -> PersistenceManager:
    """Singleton pattern for PersistenceManager."""
    global _persistence
    if _persistence is None:
        _persistence = PersistenceManager()
    return _persistence

```

### ./data_module/persistence_models.py

```python
"""
Persistence Models
Data classes for structured storage and alerting keys.
"""

from dataclasses import dataclass
from datetime import datetime
import pytz
from typing import Dict, Optional

@dataclass(frozen=True)
class AlertKey:
    """
    Structured key for duplicate alert detection.
    Frozen so it can be hashed and used as a dict key.
    """
    instrument: str
    signal_type: str
    level_ticks: int  # price normalized to ticks
    date: str         # 'YYYY-MM-DD'
    
    def __str__(self):
        return f"{self.instrument}|{self.signal_type}|{self.level_ticks}|{self.date}"

def build_alert_key(signal: Dict, tick_size: float = 0.05) -> AlertKey:
    """
    Factory to create an AlertKey from a signal dictionary.
    """
    # Use price_level (for SR/Breakout) or entry_price (for patterns)
    level = signal.get("price_level") or signal.get("entry_price") or 0.0
    
    # Normalize to ticks to avoid float rounding dupes
    if tick_size > 0:
        level_ticks = int(round(level / tick_size))
    else:
        level_ticks = int(level)
        
    # Use localized date
    ist = pytz.timezone("Asia/Kolkata")
    today = datetime.now(ist).strftime("%Y-%m-%d")
    
    return AlertKey(
        instrument=signal["instrument"],
        signal_type=signal["signal_type"],
        level_ticks=level_ticks,
        date=today,
    )

```

### ./data_module/trade_tracker.py

```python
"""
Trade Tracker Module
Tracks individual trade alerts and their outcomes in Firestore.
"""

import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pytz

from google.cloud import firestore
from google.cloud.firestore import FieldFilter
from config.settings import TIME_ZONE

logger = logging.getLogger(__name__)

class TradeTracker:
    """Tracks trades and performance stats."""
    
    def __init__(self):
        self.db = None
        self.collection_name = "trades"
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        
        if not self.project_id:
            logger.warning("⚠️ GOOGLE_CLOUD_PROJECT not set, trade tracking disabled")
            return

        try:
            self.db = firestore.Client(project=self.project_id)
        except Exception as e:
            logger.error(f"❌ Firestore init failed for TradeTracker: {str(e)}")

    def record_alert(self, signal: Dict) -> Optional[str]:
        """
        Record a new trade alert.
        Returns trade_id if successful.
        """
        if not self.db:
            return None

        try:
            trade_id = str(uuid.uuid4())
            ist = pytz.timezone(TIME_ZONE)
            now = datetime.now(ist)
            
            # Clean up signal data for storage (remove non-serializable objects)
            trade_data = {
                "trade_id": trade_id,
                "timestamp": now,
                "date": now.strftime("%Y-%m-%d"),
                "instrument": signal.get("instrument"),
                "signal_type": signal.get("signal_type"),
                "price_level": signal.get("price_level"),
                "entry_price": signal.get("entry_price"),
                "stop_loss": signal.get("stop_loss"),
                "take_profit": signal.get("take_profit"),
                "confidence": signal.get("confidence"),
                "risk_reward": signal.get("risk_reward_ratio"),
                "risk_reward": signal.get("risk_reward_ratio"),
                "description": signal.get("description"),
                "atr": signal.get("atr", 0.0),
                "status": "OPEN",  # OPEN, WIN, LOSS, BREAKEVEN
                "filters": signal.get("debug_info", {}),
                "outcome": None
            }
            
            self.db.collection(self.collection_name).document(trade_id).set(trade_data)
            logger.info(f"📝 Trade recorded: {trade_id} | {signal.get('signal_type')}")
            return trade_id
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade: {str(e)}")
            return None

    def update_outcome(self, trade_id: str, outcome: Dict) -> bool:
        """
        Update trade outcome.
        outcome = {
            "status": "WIN" | "LOSS" | "BREAKEVEN",
            "exit_price": float,
            "pnl_points": float,
            "duration_mins": float
        }
        """
        if not self.db:
            return False

        try:
            doc_ref = self.db.collection(self.collection_name).document(trade_id)
            doc_ref.update({
                "status": outcome.get("status"),
                "outcome": outcome,
                "closed_at": firestore.SERVER_TIMESTAMP
            })
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update trade outcome: {str(e)}")
            return False

    def check_open_trades(self, current_prices: Dict[str, float]) -> int:
        """
        Check all open trades and automatically close them if TP or SL is hit.
        
        Args:
            current_prices: Dict of {instrument: current_price}
            
        Returns:
            Number of trades closed
        """
        if not self.db:
            return 0

        try:
            ist = pytz.timezone(TIME_ZONE)
            now = datetime.now(ist)
            
            # Query open trades
            trades_ref = self.db.collection(self.collection_name)
            query = trades_ref.where(filter=FieldFilter("status", "==", "OPEN")).stream()
            
            closed_count = 0
            
            for doc in query:
                trade = doc.to_dict()
                trade_id = trade.get("trade_id")
                instrument = trade.get("instrument")
                entry = trade.get("entry_price")
                tp = trade.get("take_profit")
                sl = trade.get("stop_loss")
                signal_type = trade.get("signal_type", "")
                opened_at = trade.get("timestamp")
                
                # Skip if we don't have current price for this instrument
                if instrument not in current_prices:
                    continue
                
                current_price = current_prices[instrument]
                
                # Determine if LONG or SHORT
                is_long = "BULLISH" in signal_type or "SUPPORT" in signal_type or "LONG" in signal_type
                atr = trade.get("atr", 0.0)
                
                # ===========================
                # ATR TRAILING STOP LOGIC
                # ===========================
                # If trade is in profit, move SL to lock gains
                # We use 1.5x ATR for trailing (tighter than initial 2.0x usually)
                trailing_mult = 1.5
                
                if atr > 0:
                    if is_long:
                        # New potential SL = Current Price - (ATR * Mult)
                        potential_sl = current_price - (atr * trailing_mult)
                        # Only move SL UP
                        if potential_sl > sl:
                            logger.info(f"🔄 Trailing SL updated for {instrument}: {sl:.2f} -> {potential_sl:.2f}")
                            sl = potential_sl
                            # Update in DB deeply
                            doc.reference.update({"stop_loss": sl})
                    else:
                        # SHORT
                        potential_sl = current_price + (atr * trailing_mult)
                        # Only move SL DOWN
                        if potential_sl < sl:
                            logger.info(f"🔄 Trailing SL updated for {instrument}: {sl:.2f} -> {potential_sl:.2f}")
                            sl = potential_sl
                            doc.reference.update({"stop_loss": sl})
                
                outcome = None
                exit_price = None
                
                # Check if TP or SL hit
                if is_long:
                    # LONG trade
                    if current_price >= tp:
                        # Target hit - WIN
                        outcome = "WIN"
                        exit_price = tp
                    elif current_price <= sl:
                        # Stop loss hit - LOSS
                        outcome = "LOSS"
                        exit_price = sl
                else:
                    # SHORT trade
                    if current_price <= tp:
                        # Target hit - WIN
                        outcome = "WIN"
                        exit_price = tp
                    elif current_price >= sl:
                        # Stop loss hit - LOSS
                        outcome = "LOSS"
                        exit_price = sl
                
                # Update trade if outcome determined
                if outcome:
                    pnl_points = abs(exit_price - entry) if outcome == "WIN" else -abs(exit_price - entry)
                    
                    # Calculate duration
                    duration_mins = (now - opened_at).total_seconds() / 60.0 if opened_at else 0
                    
                    outcome_data = {
                        "status": outcome,
                        "exit_price": exit_price,
                        "pnl_points": pnl_points,
                        "duration_mins": duration_mins,
                        "closed_by": "AUTO"
                    }
                    
                    # Update in Firestore
                    doc_ref = self.db.collection(self.collection_name).document(trade_id)
                    doc_ref.update({
                        "status": outcome,
                        "outcome": outcome_data,
                        "closed_at": now
                    })
                    
                    closed_count += 1
                    logger.info(
                        f"✅ Trade auto-closed: {instrument} {signal_type} | "
                        f"{outcome} @ {exit_price:.2f} | P&L: {pnl_points:.2f}"
                    )
            
            return closed_count
            
        except Exception as e:
            logger.error(f"❌ Failed to check open trades: {str(e)}")
            return 0

    def get_stats(self, days: int = 7) -> Dict:
        """
        Calculate performance stats for the last N days.
        """
        if not self.db:
            return {}

        try:
            # Calculate start date (IST aware)
            ist = pytz.timezone(TIME_ZONE)
            start_date = (datetime.now(ist) - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Query trades
            trades_ref = self.db.collection(self.collection_name)
            query = trades_ref.where(filter=FieldFilter("date", ">=", start_date)).stream()
            
            total_alerts = 0
            wins = 0
            losses = 0
            by_type = {}
            
            for doc in query:
                trade = doc.to_dict()
                total_alerts += 1
                
                stype = trade.get("signal_type", "UNKNOWN")
                status = trade.get("status", "OPEN")
                
                # Stats by type
                if stype not in by_type:
                    by_type[stype] = {"count": 0, "wins": 0, "losses": 0}
                
                by_type[stype]["count"] += 1
                
                if status == "WIN":
                    wins += 1
                    by_type[stype]["wins"] += 1
                elif status == "LOSS":
                    losses += 1
                    by_type[stype]["losses"] += 1
            
            win_rate = (wins / (wins + losses)) * 100 if (wins + losses) > 0 else 0
            
            return {
                "period_days": days,
                "total_alerts": total_alerts,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "by_type": by_type
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {str(e)}")
            return {}

# Singleton
_tracker = None

def get_trade_tracker() -> TradeTracker:
    global _tracker
    if _tracker is None:
        _tracker = TradeTracker()
    return _tracker

```

## Analysis Module

### ./analysis_module/__init__.py

```python

```

### ./analysis_module/adaptive_thresholds.py

```python
"""
Adaptive Threshold Module
Adjusts RSI and ATR thresholds based on market volatility regime.
"""

import logging
from typing import Dict, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class AdaptiveThresholds:
    """
    Dynamically adjust technical thresholds based on market conditions.
    """
    
    @staticmethod
    def get_rsi_thresholds(vix: float = None, atr_percentile: float = None) -> Tuple[int, int]:
        """
        Get adaptive RSI thresholds based on volatility.
        
        Args:
            vix: India VIX value (if available)
            atr_percentile: ATR percentile (0-100)
        
        Returns:
            (rsi_long_threshold, rsi_short_threshold)
        """
        # Default values (normal market)
        rsi_long = 60
        rsi_short = 40
        
        # VIX-based adjustment (preferred)
        if vix is not None:
            if vix < 12:
                # Low volatility = sideways = tighter bands
                rsi_long = 55
                rsi_short = 45
                logger.debug(f"📊 Adaptive RSI: VIX {vix:.1f} (Low Volatility) → Tighter ({rsi_short}/{rsi_long})")
            elif vix > 18:
                # High volatility = trending = wider bands
                rsi_long = 65
                rsi_short = 35
                logger.debug(f"📊 Adaptive RSI: VIX {vix:.1f} (High Volatility) → Wider ({rsi_short}/{rsi_long})")
            else:
                # Normal
                logger.debug(f"📊 Adaptive RSI: VIX {vix:.1f} (Normal) → Default ({rsi_short}/{rsi_long})")
        
        # ATR percentile fallback (if VIX unavailable)
        elif atr_percentile is not None:
            if atr_percentile < 30:
                # Low volatility
                rsi_long = 55
                rsi_short = 45
                logger.debug(f"📊 Adaptive RSI: ATR %ile {atr_percentile:.1f} (Low Vol) → Tighter")
            elif atr_percentile > 70:
                # High volatility
                rsi_long = 65
                rsi_short = 35
                logger.debug(f"📊 Adaptive RSI: ATR %ile {atr_percentile:.1f} (High Vol) → Wider")
        
        return rsi_long, rsi_short
    
    @staticmethod
    def get_atr_threshold(df: pd.DataFrame, atr_period: int = 14) -> float:
        """
        Get adaptive ATR threshold as percentile of recent ATR values.
        
        Returns: ATR value at 60th percentile (tradeable volatility)
        """
        if df is None or len(df) < atr_period + 60:
            return 0.0
        
        # Calculate ATR history
        if "atr" not in df.columns:
            logger.warning("⚠️ ATR column not found in dataframe")
            return 0.0
            
        atr_history = df["atr"].iloc[-(atr_period + 60):-1]
        
        # Use 60th percentile as threshold
        atr_threshold = atr_history.quantile(0.60)
        
        logger.debug(f"📊 Adaptive ATR Threshold: {atr_threshold:.2f} (60th %ile)")
        return atr_threshold
    
    @staticmethod
    def calculate_atr_percentile(df: pd.DataFrame, current_atr: float, lookback: int = 60) -> float:
        """Calculate what percentile current ATR is in recent history."""
        if df is None or len(df) < lookback:
            return 50.0  # Default to median
        
        if "atr" not in df.columns:
            logger.warning("⚠️ ATR column not found in dataframe")
            return 50.0
            
        atr_history = df["atr"].iloc[-lookback:]
        percentile = (atr_history < current_atr).sum() / len(atr_history) * 100
        
        return percentile
    
    @staticmethod
    def is_market_volatile(vix: float = None, atr_percentile: float = None) -> bool:
        """
        Determine if market is in high volatility regime.
        
        Returns:
            True if volatile (VIX > 18 or ATR percentile > 70)
        """
        if vix is not None and vix > 18:
            return True
        if atr_percentile is not None and atr_percentile > 70:
            return True
        return False
    
    @staticmethod
    def is_market_choppy(vix: float = None, atr_percentile: float = None) -> bool:
        """
        Determine if market is in low volatility / choppy regime.
        
        Returns:
            True if choppy (VIX < 12 or ATR percentile < 30)
        """
        if vix is not None and vix < 12:
            return True
        if atr_percentile is not None and atr_percentile < 30:
            return True
        return False

```

### ./analysis_module/combo_signals.py

```python
"""
MACD + RSI + Bollinger Bands Combo Signal Evaluator
Analyzes confluence of multiple indicators for signal strength.
"""

import logging
from typing import Dict
import pandas as pd

logger = logging.getLogger(__name__)


class MACDRSIBBCombo:
    """
    Evaluates signal strength based on MACD, RSI, and Bollinger Bands confluence.
    
    Strength Levels:
        - STRONG: All 3 conditions met + extreme RSI
        - MEDIUM: 2 out of 3 conditions met
        - WEAK: Only 1 condition met
        - INVALID: No conditions met or opposing signals
    """
    
    def __init__(self):
        logger.info("📊 MACDRSIBBCombo initialized")
    
    def evaluate_signal(
        self, 
        df: pd.DataFrame, 
        direction_bias: str,  # From EMA crossover or pattern direction
        technical_context: Dict
    ) -> Dict:
        """
        Evaluate signal strength based on indicator confluence.
        
        Args:
            df: OHLCV DataFrame with indicators
            direction_bias: "BULLISH" or "BEARISH" from EMA crossover or pattern
            technical_context: Dict with MACD, RSI, BB data
        
        Returns:
            {
                "strength": str,  # "STRONG", "MEDIUM", "WEAK", "INVALID"
                "score": int,     # 0-3 (number of conditions met)
                "conditions": {
                    "bb_favorable": bool,
                    "rsi_favorable": bool,
                    "macd_favorable": bool
                },
                "bb_position": float,  # 0.0 to 1.0
                "details": str  # Human-readable explanation
            }
        """
        try:
            current_price = df['close'].iloc[-1]
            
            # Extract indicators from context
            macd_data = technical_context.get("macd", {})
            rsi = technical_context.get("rsi_5", 50)
            bb_upper = technical_context.get("bb_upper", 0)
            bb_lower = technical_context.get("bb_lower", 0)
            
            # Get previous RSI for trend detection
            rsi_prev = 50
            if len(df) >= 2:
                # Calculate RSI series if needed
                if 'rsi' not in df.columns:
                    from config.settings import RSI_PERIOD
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                
                if not df['rsi'].isna().iloc[-2]:
                    rsi_prev = df['rsi'].iloc[-2]
            
            # Calculate Bollinger Band position (0 = lower band, 1 = upper band)
            bb_position = 0.5
            if bb_upper > bb_lower > 0:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                bb_position = max(0.0, min(1.0, bb_position))  # Clamp to [0, 1]
            
            # Evaluate conditions
            conditions = {}
            conditions_met = 0
            
            if direction_bias == "BULLISH" or direction_bias == "LONG":
                # Condition 1: Price in lower 35% of Bollinger Bands (oversold zone)
                conditions["bb_favorable"] = bb_position < 0.35
                
                # Condition 2: RSI < 40 and RISING (momentum building)
                conditions["rsi_favorable"] = rsi < 40 and rsi > rsi_prev
                
                # Condition 3:MACD histogram > 0 OR bullish crossover
                conditions["macd_favorable"] = (
                    macd_data.get("histogram", 0) > 0 or 
                    macd_data.get("crossover") == "BULLISH"
                )
                
            elif direction_bias == "BEARISH" or direction_bias == "SHORT":
                # Condition 1: Price in upper 35% of Bollinger Bands (overbought zone)
                conditions["bb_favorable"] = bb_position > 0.65
                
                # Condition 2: RSI > 60 and FALLING (momentum weakening)
                conditions["rsi_favorable"] = rsi > 60 and rsi < rsi_prev
                
                # Condition 3: MACD histogram < 0 OR bearish crossover
                conditions["macd_favorable"] = (
                    macd_data.get("histogram", 0) < 0 or 
                    macd_data.get("crossover") == "BEARISH"
                )
            
            else:  # NEUTRAL
                return {
                    "strength": "INVALID",
                    "score": 0,
                    "conditions": {},
                    "bb_position": bb_position,
                    "details": "No directional bias"
                }
            
            # Count conditions met
            conditions_met = sum(conditions.values())
            
            # Determine strength
            strength = "INVALID"
            details = ""
            
            # Normalize direction for checks
            is_bullish = direction_bias in ["BULLISH", "LONG"]
            
            if is_bullish:
                if conditions_met >= 3 and rsi < 30:
                    strength = "STRONG"
                    details = f"All conditions met + RSI extreme ({rsi:.1f})"
                elif conditions_met >= 2:
                    strength = "MEDIUM"
                    details = f"{conditions_met}/3 conditions met"
                elif conditions_met == 1:
                    strength = "WEAK"
                    details = f"Only {conditions_met}/3 condition met"
                else:
                    strength = "INVALID"
                    details = "No conditions met"
            
            else:  # BEARISH/SHORT
                if conditions_met >= 3 and rsi > 70:
                    strength = "STRONG"
                    details = f"All conditions met + RSI extreme ({rsi:.1f})"
                elif conditions_met >= 2:
                    strength = "MEDIUM"
                    details = f"{conditions_met}/3 conditions met"
                elif conditions_met == 1:
                    strength = "WEAK"
                    details = f"Only {conditions_met}/3 condition met"
                else:
                    strength = "INVALID"
                    details = "No conditions met"
            
            # Log the evaluation
            logger.info(
                f"📊 Combo Signal: {direction_bias} | {strength} | "
                f"Score: {conditions_met}/3 | BB: {bb_position:.2f} | RSI: {rsi:.1f} | "
                f"MACD Hist: {macd_data.get('histogram', 0):.2f}"
            )
            
            return {
                "strength": strength,
                "score": conditions_met,
                "conditions": conditions,
                "bb_position": round(bb_position, 2),
                "details": details
            }
        
        except Exception as e:
            logger.error(f"Error evaluating combo signal: {e}")
            return {
                "strength": "INVALID",
                "score": 0,
                "conditions": {},
                "bb_position": 0.5,
                "details": f"Error: {str(e)}"
            }

```

### ./analysis_module/confluence_detector.py

```python
"""
Confluence Detection Module
Identifies when multiple technical levels converge at the same price point.
Expert's core methodology for high-probability setups.
"""

import logging
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
    rsi_divergence: str = "NONE"
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    ema_50: float = 0.0
    r1_fib: float = 0.0
    s1_fib: float = 0.0
    r2_fib: float = 0.0
    s2_fib: float = 0.0
    confluence_zones: List[Dict] = None


def detect_confluence(
    price: float, 
    levels: TechnicalLevels,
    higher_tf_context: Dict,
    tolerance_pct: float = 0.002  # 0.2% tolerance (±3-5 points for Nifty)
) -> Dict:
    """
    Detect confluence of multiple technical levels at current price.
    
    Expert's methodology: Identify when 2+ key levels converge within ±3 points.
    
    Args:
        price: Current price to check
        levels: TechnicalLevels object with all key levels
        higher_tf_context: Context with EMAs, VWAP, etc.
        tolerance_pct: Percentage tolerance for "near" (default 0.2% = ~5 points for 25,000)
        
    Returns:
        Dict with:
            - confluence_count: Number of levels near price
            - level_names: List of confluent level names
            - confluence_score: 0-100 score based on quality
            - is_high_probability: True if 2+ levels converge
    """
    tolerance = price * tolerance_pct
    confluent_levels = []
    
    # Check all key levels
    level_checks = {
        'PDH': levels.pdh,
        'PDL': levels.pdl,
        'Fib_R1': levels.r1_fib,
        'Fib_S1': levels.s1_fib,
        'Fib_R2': levels.r2_fib,
        'Fib_S2': levels.s2_fib,
        'Pivot': levels.pivot,
        'EMA20_5m': higher_tf_context.get('ema_20_5m', 0),
        'EMA50_5m': higher_tf_context.get('ema_50_5m', 0),
        'EMA50_15m': higher_tf_context.get('ema_50_15m', 0),
        'VWAP': higher_tf_context.get('vwap_5m', 0),
        'BB_Upper': higher_tf_context.get('bb_upper_5m', 0),
        'BB_Middle': higher_tf_context.get('bb_middle_5m', 0),
        'BB_Lower': higher_tf_context.get('bb_lower_5m', 0),
    }
    
    for name, level in level_checks.items():
        if level > 0 and abs(price - level) <= tolerance:
            confluent_levels.append({
                'name': name,
                'level': level,
                'distance': abs(price - level),
                'distance_pct': abs(price - level) / price * 100
            })
    
    # Calculate confluence score
    confluence_count = len(confluent_levels)
    
    # Base score
    if confluence_count == 0:
        confluence_score = 0
    elif confluence_count == 1:
        confluence_score = 30
    elif confluence_count == 2:
        confluence_score = 70  # Expert's threshold
    elif confluence_count == 3:
        confluence_score = 90
    else:
        confluence_score = 100
    
    # Bonus for key combinations (Expert's favorites)
    level_names = [l['name'] for l in confluent_levels]
    
    # PDH + Fib R1 (resistance confluence)
    if 'PDH' in level_names and 'Fib_R1' in level_names:
        confluence_score = min(100, confluence_score + 10)
        
    # PDL + Fib S1 (support confluence)
    if 'PDL' in level_names and 'Fib_S1' in level_names:
        confluence_score = min(100, confluence_score + 10)
        
    # EMA20 + Fib Level (dynamic support/resistance)
    if 'EMA20_5m' in level_names and any('Fib' in n for n in level_names):
        confluence_score = min(100, confluence_score + 10)
        
    # BB extreme + Fib level (oversold/overbought confluence)
    if ('BB_Upper' in level_names or 'BB_Lower' in level_names) and any('Fib' in n for n in level_names):
        confluence_score = min(100, confluence_score + 10)
    
    is_high_probability = confluence_count >= 2
    
    if confluence_count >= 2:
        logger.info(
            f"🎯 CONFLUENCE DETECTED | {confluence_count} levels | "
            f"Price: {price:.2f} | Levels: {', '.join(level_names)}"
        )
    
    return {
        'confluence_count': confluence_count,
        'level_names': level_names,
        'confluent_levels': confluent_levels,
        'confluence_score': confluence_score,
        'is_high_probability': is_high_probability,
        'tolerance_used': tolerance
    }

```

### ./analysis_module/manipulation_guard.py

```python
"""
Manipulation Guard (Circuit Breaker)
Protects against Flash Crashes, Freak Trades, and Expiry Manipulation.
"""

import logging
from datetime import datetime, time
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

from config.settings import (
    MAX_1MIN_MOVE_PCT,
    EXPIRY_STOP_TIME,
    VIX_PANIC_LEVEL,
    VIX_LOW_LEVEL,
    TIME_ZONE
)

logger = logging.getLogger(__name__)

class CircuitBreaker:
    def __init__(self):
        self.triggered = False
        self.trigger_reason = None
        self.trigger_time = None
        self.pause_duration = 0
        self.last_vix = 0.0

    def check_market_integrity(self, df_5m: pd.DataFrame, current_price: float, instrument: str = "NIFTY 50") -> Tuple[bool, str]:
        """
        Run all safety checks.
        Returns: (is_safe, reason)
        """
        
        # 1. Check if Breaker already tripped
        if self.triggered:
            elapsed = (datetime.now() - self.trigger_time).total_seconds() / 60
            if elapsed < self.pause_duration:
                return False, f"Circuit Breaker Active ({self.trigger_reason}) - {int(self.pause_duration - elapsed)}m remaining"
            else:
                self._reset_breaker()

        # 2. Flash Crash / Velocity Check (1-minute equivalent using last 5m candle limit)
        # Note: Ideally we check tick data or 1m data. Using 5m rapid move proxy.
        if not df_5m.empty:
            last_candle = df_5m.iloc[-1]
            high = last_candle['high']
            low = last_candle['low']
            open_p = last_candle['open']
            
            # Use High-Low range as proxy for volatility/velocity
            move_pct = ((high - low) / open_p) * 100
            
            # If a single 5m candle moves > 2x the 1min limit (approx), it's a crash/spike
            if move_pct > (MAX_1MIN_MOVE_PCT * 2.5): 
                self._trip_breaker("Flash Move Detected", 15)
                return False, f"Flash Crash Protection: {move_pct:.2f}% move in 5m"

        # 3. Expiry Day Gamma Guard
        if self._is_expiry_danger_zone(instrument):
             return False, "Expiry Gamma Guard Active (Post 2:00 PM)"

        # 4. Freak Trade Filter (Wick check)
        # If current price is far from last close but within candle? (Implied in Flash check)

        return True, "Market Normal"

    def _is_expiry_danger_zone(self, instrument: str) -> bool:
        """Check if it's Tuesday (Nifty Expiry) and past the Stop Time."""
        now = datetime.now()
        
        # NIFTY 50 Expiry = Tuesday (Weekday 1)
        # BANKNIFTY Expiry = Wednesday (Weekday 2) -> Future improvement
        
        is_expiry_day = False
        if "NIFTY" in instrument and "BANK" not in instrument:
             is_expiry_day = now.weekday() == 1  # 1 = Tuesday
        
        if not is_expiry_day:
            return False
            
        # Parse Stop Time
        stop_hour, stop_min = map(int, EXPIRY_STOP_TIME.split(":"))
        if now.time() >= time(stop_hour, stop_min):
            return True
            
        return False

    def _trip_breaker(self, reason: str, duration_mins: int):
        self.triggered = True
        self.trigger_reason = reason
        self.trigger_time = datetime.now()
        self.pause_duration = duration_mins
        logger.warning(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}. Pausing for {duration_mins} mins.")

    def _reset_breaker(self):
        self.triggered = False
        self.trigger_reason = None
        self.trigger_time = None
        logger.info("✅ Circuit Breaker Reset. Resuming operations.")
        
    def check_vix(self, vix_value: float) -> str:
        """Return market mode based on VIX"""
        if vix_value > VIX_PANIC_LEVEL:
            return "PANIC"
        if vix_value < VIX_LOW_LEVEL:
            return "DEAD"
        return "NORMAL"

```

### ./analysis_module/market_state_engine.py

```python
"""
Market State Engine
Determines if market conditions are suitable for scalping

States:
- CHOPPY: No trading (capital protection)
- TRANSITION: Selective trading (high-momentum only)
- EXPANSIVE: Normal trading (all strategies active)
"""

import logging
import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """Market state for scalping suitability."""
    CHOPPY = "CHOPPY"
    TRANSITION = "TRANSITION"
    EXPANSIVE = "EXPANSIVE"


class MarketStateEngine:
    """
    Evaluates market state for scalping suitability.
    
    Uses multiple metrics to determine if market is:
    - CHOPPY: Overlapping, non-directional (no trades)
    - TRANSITION: Breaking from compression (selective trades)
    - EXPANSIVE: Strong directional movement (full trading)
    """
    
    def __init__(
        self,
        choppy_range_threshold: float = 20.0,
        choppy_wick_ratio: float = 0.55,
        choppy_vwap_crosses: int = 4,
        expansion_threshold: float = 12.0,
        expansive_range_threshold: float = 25.0
    ):
        """
        Initialize Market State Engine.
        
        Args:
            choppy_range_threshold: Max 6-candle range for CHOPPY (points)
            choppy_wick_ratio: Min avg wick ratio for CHOPPY
            choppy_vwap_crosses: Min VWAP crosses for CHOPPY
            expansion_threshold: Min follow-through for expansion (points)
            expansive_range_threshold: Min range for EXPANSIVE state (points)
        """
        self.choppy_range_threshold = choppy_range_threshold
        self.choppy_wick_ratio = choppy_wick_ratio
        self.choppy_vwap_crosses = choppy_vwap_crosses
        self.expansion_threshold = expansion_threshold
        self.expansive_range_threshold = expansive_range_threshold
        
        logger.info("✅ Market State Engine initialized")
    
    def evaluate_state(
        self,
        df: pd.DataFrame,
        vwap_series: pd.Series = None
    ) -> Dict:
        """
        Evaluate current market state.
        
        Args:
            df: OHLCV dataframe (last 20+ candles)
            vwap_series: Optional VWAP values
            
        Returns:
            {
                "state": MarketState,
                "confidence": float (0-1),
                "reasons": List[str],
                "metrics": Dict[str, any]
            }
        """
        if len(df) < 10:
            logger.warning("Insufficient data for state evaluation")
            return self._default_state()
        
        # Check CHOPPY conditions (any 2 of 4)
        choppy_score, choppy_metrics = self._check_choppy_conditions(df, vwap_series)
        
        if choppy_score >= 2:
            reasons = self._get_choppy_reasons(choppy_metrics)
            return {
                "state": MarketState.CHOPPY,
                "confidence": choppy_score / 4.0,
                "reasons": reasons,
                "metrics": choppy_metrics
            }
        
        # Check EXPANSIVE conditions
        expansive_score, expansive_metrics = self._check_expansive_conditions(df)
        
        if expansive_score >= 2:
            reasons = self._get_expansive_reasons(expansive_metrics)
            return {
                "state": MarketState.EXPANSIVE,
                "confidence": expansive_score / 3.0,
                "reasons": reasons,
                "metrics": expansive_metrics
            }
        
        # Default to TRANSITION (between CHOPPY and EXPANSIVE)
        return {
            "state": MarketState.TRANSITION,
            "confidence": 0.5,
            "reasons": ["Market breaking from compression"],
            "metrics": {
                "choppy_score": choppy_score,
                "expansive_score": expansive_score
            }
        }
    
    def _check_choppy_conditions(
        self,
        df: pd.DataFrame,
        vwap_series: pd.Series = None
    ) -> Tuple[int, Dict]:
        """
        Check for CHOPPY market conditions.
        
        Returns:
            (score, metrics) where score is 0-4 based on met conditions
        """
        score = 0
        metrics = {}
        
        # Condition 1: Range Compression
        # Last 6 candles have total range < 20 points
        last_6 = df.tail(6)
        total_range = last_6['high'].max() - last_6['low'].min()
        metrics["range_6_candles"] = total_range
        
        if total_range < self.choppy_range_threshold:
            score += 1
            metrics["range_compressed"] = True
        else:
            metrics["range_compressed"] = False
        
        # Condition 2: Wick Dominance
        # Average wick-to-candle ratio > 55%
        avg_wick_ratio = self._calculate_wick_ratio(last_6)
        metrics["avg_wick_ratio"] = avg_wick_ratio
        
        if avg_wick_ratio > self.choppy_wick_ratio:
            score += 1
            metrics["wick_dominant"] = True
        else:
            metrics["wick_dominant"] = False
        
        # Condition 3: VWAP Magnet
        # Price crosses VWAP 4+ times in last 10 candles
        if vwap_series is not None and len(vwap_series) >= 10:
            vwap_crosses = self._count_vwap_crosses(df.tail(10), vwap_series.tail(10))
            metrics["vwap_crosses"] = vwap_crosses
            
            if vwap_crosses >= self.choppy_vwap_crosses:
                score += 1
                metrics["vwap_magnet"] = True
            else:
                metrics["vwap_magnet"] = False
        else:
            metrics["vwap_crosses"] = 0
            metrics["vwap_magnet"] = False
        
        # Condition 4: Expansion Failure
        # No sustained follow-through after breaks
        has_expansion = self._check_expansion_follow_through(df)
        metrics["has_expansion"] = has_expansion
        
        if not has_expansion:
            score += 1
            metrics["expansion_failed"] = True
        else:
            metrics["expansion_failed"] = False
        
        return score, metrics
    
    def _check_expansive_conditions(self, df: pd.DataFrame) -> Tuple[int, Dict]:
        """
        Check for EXPANSIVE market conditions.
        
        Returns:
            (score, metrics) where score is 0-3 based on met conditions
        """
        score = 0
        metrics = {}
        
        last_10 = df.tail(10)
        
        # Condition 1: Large candle bodies (low wick ratio)
        avg_wick_ratio = self._calculate_wick_ratio(last_10)
        metrics["avg_wick_ratio"] = avg_wick_ratio
        
        if avg_wick_ratio < 0.35:  # Bodies > 65% of candle
            score += 1
            metrics["large_bodies"] = True
        else:
            metrics["large_bodies"] = False
        
        # Condition 2: Directional follow-through
        has_follow_through = self._check_directional_follow_through(last_10)
        metrics["has_follow_through"] = has_follow_through
        
        if has_follow_through:
            score += 1
        
        # Condition 3: Sustained range expansion
        total_range = last_10['high'].max() - last_10['low'].min()
        metrics["range_10_candles"] = total_range
        
        if total_range >= self.expansive_range_threshold:
            score += 1
            metrics["range_expanded"] = True
        else:
            metrics["range_expanded"] = False
        
        return score, metrics
    
    def _calculate_wick_ratio(self, df: pd.DataFrame) -> float:
        """
        Calculate average wick-to-candle ratio.
        
        Wick ratio = (upper_wick + lower_wick) / total_range
        """
        ratios = []
        
        for _, candle in df.iterrows():
            high = candle['high']
            low = candle['low']
            open_price = candle['open']
            close = candle['close']
            
            total_range = high - low
            if total_range == 0:
                continue
            
            body_high = max(open_price, close)
            body_low = min(open_price, close)
            
            upper_wick = high - body_high
            lower_wick = body_low - low
            
            wick_ratio = (upper_wick + lower_wick) / total_range
            ratios.append(wick_ratio)
        
        return np.mean(ratios) if ratios else 0.0
    
    def _count_vwap_crosses(
        self,
        df: pd.DataFrame,
        vwap_series: pd.Series
    ) -> int:
        """Count how many times price crosses VWAP."""
        crosses = 0
        
        for i in range(1, len(df)):
            prev_close = df.iloc[i-1]['close']
            curr_close = df.iloc[i]['close']
            prev_vwap = vwap_series.iloc[i-1]
            curr_vwap = vwap_series.iloc[i]
            
            # Check if price crossed VWAP
            prev_above = prev_close > prev_vwap
            curr_above = curr_close > curr_vwap
            
            if prev_above != curr_above:
                crosses += 1
        
        return crosses
    
    def _check_expansion_follow_through(self, df: pd.DataFrame) -> bool:
        """
        Check if recent breaks show follow-through expansion.
        
        Returns True if any recent break expanded 12-15 points within 2 candles.
        """
        if len(df) < 5:
            return False
        
        # Look for breaks in last 5 candles
        for i in range(len(df) - 4, len(df) - 1):
            candle = df.iloc[i]
            prev_candles = df.iloc[max(0, i-5):i]
            
            # Check if this candle broke recent high/low
            broke_high = candle['high'] > prev_candles['high'].max()
            broke_low = candle['low'] < prev_candles['low'].min()
            
            if broke_high or broke_low:
                # Check follow-through in next 1-2 candles
                next_candles = df.iloc[i+1:min(i+3, len(df))]
                
                if broke_high:
                    extension = next_candles['high'].max() - candle['high']
                else:
                    extension = candle['low'] - next_candles['low'].min()
                
                if extension >= self.expansion_threshold:
                    return True
        
        return False
    
    def _check_directional_follow_through(self, df: pd.DataFrame) -> bool:
        """
        Check if candles show sustained directional movement.
        
        Returns True if 3+ consecutive candles move in same direction.
        """
        if len(df) < 3:
            return False
        
        # Check bullish sequences
        bullish_count = 0
        bearish_count = 0
        
        for _, candle in df.iterrows():
            if candle['close'] > candle['open']:
                bullish_count += 1
                bearish_count = 0
            else:
                bearish_count += 1
                bullish_count = 0
            
            if bullish_count >= 3 or bearish_count >= 3:
                return True
        
        return False
    
    def _get_choppy_reasons(self, metrics: Dict) -> List[str]:
        """Generate human-readable reasons for CHOPPY state."""
        reasons = []
        
        if metrics.get("range_compressed"):
            reasons.append(f"Range compressed: {metrics['range_6_candles']:.1f}pts")
        
        if metrics.get("wick_dominant"):
            reasons.append(f"Wick dominance: {metrics['avg_wick_ratio']:.0%}")
        
        if metrics.get("vwap_magnet"):
            reasons.append(f"VWAP magnet: {metrics['vwap_crosses']} crosses")
        
        if metrics.get("expansion_failed"):
            reasons.append("No follow-through expansion")
        
        return reasons
    
    def _get_expansive_reasons(self, metrics: Dict) -> List[str]:
        """Generate human-readable reasons for EXPANSIVE state."""
        reasons = []
        
        if metrics.get("large_bodies"):
            reasons.append(f"Large bodies (wick: {metrics['avg_wick_ratio']:.0%})")
        
        if metrics.get("has_follow_through"):
            reasons.append("Directional follow-through confirmed")
        
        if metrics.get("range_expanded"):
            reasons.append(f"Range expansion: {metrics['range_10_candles']:.1f}pts")
        
        return reasons
    
    def _default_state(self) -> Dict:
        """Return default TRANSITION state for edge cases."""
        return {
            "state": MarketState.TRANSITION,
            "confidence": 0.3,
            "reasons": ["Insufficient data for full evaluation"],
            "metrics": {}
        }

```

### ./analysis_module/option_chain_analyzer.py

```python

import logging
from typing import Dict, Optional, List, Tuple

logger = logging.getLogger(__name__)

class OptionChainAnalyzer:
    """
    Analyzes option chain data to extract key metrics like PCR, Max Pain, and S/R levels.
    """
    
    def calculate_pcr(self, option_data: Dict) -> Optional[float]:
        """
        Calculate Put-Call Ratio (Total Put OI / Total Call OI).
        """
        try:
            total_call_oi = 0
            total_put_oi = 0
            
            records = option_data.get('records', {})
            data = records.get('data', [])
            
            for item in data:
                if 'CE' in item:
                    total_call_oi += item['CE'].get('openInterest', 0)
                if 'PE' in item:
                    total_put_oi += item['PE'].get('openInterest', 0)
            
            if total_call_oi == 0:
                logger.warning("OptionChainAnalyzer: Total Call OI is 0")
                return None
            
            pcr = total_put_oi / total_call_oi
            return round(pcr, 4)
            
        except Exception as e:
            logger.error(f"Error calculating PCR: {e}")
            return None

    def calculate_max_pain(self, option_data: Dict) -> Optional[float]:
        """
        Calculate Max Pain theory strike price.
        Max Pain is the strike where option writers lose the least amount of money.
        """
        try:
            records = option_data.get('records', {})
            data = records.get('data', [])
            
            strikes = []
            call_oi = {}
            put_oi = {}
            
            for item in data:
                strike = item['strikePrice']
                strikes.append(strike)
                call_oi[strike] = item.get('CE', {}).get('openInterest', 0)
                put_oi[strike] = item.get('PE', {}).get('openInterest', 0)
            
            if not strikes:
                return None
                
            # Sort strikes to ensure correct iteration
            strikes.sort()
            
            pain_values = {}
            
            for strike in strikes:
                current_pain = 0
                
                # Call Pain: (Spot - Strike) * OI for In-The-Money Calls (Spot > Strike)
                # If expiry is at 'strike', calls below 'strike' are ITM
                # Loss = (Strike_Expiry - Strike_Option) * OI
                # Here we assume expiry is at 'strike' to calculate pain AT that level
                
                # Pain if market expires at 'strike':
                
                # 1. Call Writers lose if Strike > Call_Strike
                # Loss = (Strike - Call_Strike) * Call_OI
                for s in strikes:
                    if s < strike:
                         current_pain += (strike - s) * call_oi.get(s, 0)
                         
                # 2. Put Writers lose if Strike < Put_Strike
                # Loss = (Put_Strike - Strike) * Put_OI
                for s in strikes:
                    if s > strike:
                        current_pain += (s - strike) * put_oi.get(s, 0)
                        
                pain_values[strike] = current_pain
                
            # Find strike with minimum pain
            max_pain_strike = min(pain_values, key=pain_values.get)
            return max_pain_strike
            
        except Exception as e:
            logger.error(f"Error calculating Max Pain: {e}")
            return None

    def calculate_atm_iv(self, option_data: Dict, spot_price: float) -> Optional[float]:
        """
        Calculate average Implied Volatility (IV) for ATM strikes.
        Returns average of Call IV and Put IV for the strike closest to spot price.
        """
        try:
            records = option_data.get('records', {})
            data = records.get('data', [])
            
            if not data or spot_price == 0:
                return None
                
            # Find ATM strike (closest to spot)
            atm_strike_data = min(data, key=lambda x: abs(x['strikePrice'] - spot_price))
            
            iv_sum = 0
            count = 0
            
            if 'CE' in atm_strike_data:
                iv = atm_strike_data['CE'].get('impliedVolatility', 0)
                if iv > 0:
                    iv_sum += iv
                    count += 1
                    
            if 'PE' in atm_strike_data:
                iv = atm_strike_data['PE'].get('impliedVolatility', 0)
                if iv > 0:
                    iv_sum += iv
                    count += 1
            
            if count == 0:
                return None
                
            return round(iv_sum / count, 2)
            
        except Exception as e:
            logger.error(f"Error calculating ATM IV: {e}")
            return None

    def analyze_oi_change(self, option_data: Dict, spot_price: float) -> Dict:
        """
        Analyze Change in Open Interest to determine sentiment.
        Calculates net OI change for strikes within 5% of spot.
        """
        try:
            records = option_data.get('records', {})
            data = records.get('data', [])
            
            if not data or spot_price <= 0:
                return {}
            
            # Filter for strikes within 5% range to focus on relevant activity
            relevant_data = [
                d for d in data 
                if abs(d['strikePrice'] - spot_price) < (spot_price * 0.05)
            ]
            
            total_call_change = 0
            total_put_change = 0
            
            for item in relevant_data:
                if 'CE' in item:
                    total_call_change += item['CE'].get('changeinOpenInterest', 0)
                if 'PE' in item:
                    total_put_change += item['PE'].get('changeinOpenInterest', 0)
            
            # Determine sentiment based on net flow
            # Call OI Increase > Put OI Increase -> Bearish (Resistance building)
            # Put OI Increase > Call OI Increase -> Bullish (Support building)
            
            sentiment = "NEUTRAL"
            difference = total_put_change - total_call_change
            
            # Significant difference threshold (e.g., 50k contracts)
            # Adjust based on instrument volume, but relative comparison is safer
            
            if total_call_change > 0 and total_put_change > 0:
                if total_put_change > (total_call_change * 1.5):
                    sentiment = "BULLISH"
                elif total_call_change > (total_put_change * 1.5):
                    sentiment = "BEARISH"
            
            # Handling unwinding scenarios (negative OI change)
            elif total_call_change < 0 and total_put_change < 0:
                 if abs(total_call_change) > abs(total_put_change) * 1.5:
                     sentiment = "BULLISH_UNWINDING" # Short covering
                 elif abs(total_put_change) > abs(total_call_change) * 1.5:
                     sentiment = "BEARISH_UNWINDING" # Long unwinding

            return {
                "total_call_change": total_call_change,
                "total_put_change": total_put_change,
                "sentiment": sentiment,
                "net_change_diff": difference
            }
            
        except Exception as e:
            logger.error(f"Error analyzing OI change: {e}")
            return {}

    def get_key_strikes(self, option_data: Dict) -> Dict:
        """
        Identify strikes with highest Open Interest for Support and Resistance.
        """
        try:
            records = option_data.get('records', {})
            data = records.get('data', [])
            
            call_oi_map = {}
            put_oi_map = {}
            
            for item in data:
                strike = item['strikePrice']
                if 'CE' in item:
                    call_oi_map[strike] = item['CE'].get('openInterest', 0)
                if 'PE' in item:
                    put_oi_map[strike] = item['PE'].get('openInterest', 0)
            
            # Find max OI strikes
            max_call_oi_strike = max(call_oi_map, key=call_oi_map.get) if call_oi_map else 0
            max_put_oi_strike = max(put_oi_map, key=put_oi_map.get) if put_oi_map else 0
            
            # Get top 3 levels
            top_calls = sorted(call_oi_map.items(), key=lambda x: x[1], reverse=True)[:3]
            top_puts = sorted(put_oi_map.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                "max_call_oi_strike": max_call_oi_strike,
                "max_put_oi_strike": max_put_oi_strike,
                "resistance_levels": [s for s, oi in top_calls],
                "support_levels": [s for s, oi in top_puts],
                "max_call_oi": call_oi_map.get(max_call_oi_strike, 0),
                "max_put_oi": put_oi_map.get(max_put_oi_strike, 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting key strikes: {e}")
            return {}

if __name__ == "__main__":
    # Integration test with Fetcher
    try:
        from data_module.option_chain_fetcher import OptionChainFetcher
        
        logging.basicConfig(level=logging.INFO)
        fetcher = OptionChainFetcher()
        analyzer = OptionChainAnalyzer()
        
        print("Fetching data...")
        data = fetcher.fetch_option_chain("NIFTY")
        
        if data:
            print("\n--- Analysis Results ---")
            
            pcr = analyzer.calculate_pcr(data)
            print(f"PCR: {pcr}")
            
            max_pain = analyzer.calculate_max_pain(data)
            print(f"Max Pain: {max_pain}")
            
            levels = analyzer.get_key_strikes(data)
            print(f"Max Call OI (Res): {levels['max_call_oi_strike']}")
            print(f"Max Put OI (Sup): {levels['max_put_oi_strike']}")
            print(f"Top 3 Resistances: {levels['resistance_levels']}")
            print(f"Top 3 Supports: {levels['support_levels']}")
            
        else:
            print("Failed to fetch data for analysis")
            
    except ImportError:
        print("Please run this from the project root to test imports")

```

### ./analysis_module/signal_pipeline.py

```python
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
    USE_COMBO_SIGNALS,  # NEW: Combo strategy flag
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
    6. MACD+RSI+BB Combo Validation (NEW)
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
        
        # NEW: Initialize Combo Signal Evaluator
        self.combo_evaluator = None
        if USE_COMBO_SIGNALS:
            try:
                from analysis_module.combo_signals import MACDRSIBBCombo
                self.combo_evaluator = MACDRSIBBCombo()
                logger.info("✅ MACD+RSI+BB Combo evaluator initialized")
            except Exception as e:
                logger.warning(f"⚠️ Combo evaluator failed to initialize: {e}")


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

        # 8. MACD + RSI + BB Combo Signal Evaluation (NEW)
        # ------------------------------------------------
        if USE_COMBO_SIGNALS and self.combo_evaluator:
            try:
                # Determine signal direction
                direction = "BULLISH" if is_bullish else "BEARISH"
                
                # Get dataframe from context
                df = analysis_context.get("df_5m") or analysis_context.get("df")
                
                if df is not None and len(df) >= 50:
                    # Calculate MACD if not in context
                    macd_data = htf.get("macd")
                    if not macd_data:
                        # Import TechnicalAnalyzer to calculate MACD
                        from analysis_module.technical import TechnicalAnalyzer
                        analyzer = TechnicalAnalyzer("TEMP")
                        macd_data = analyzer._calculate_macd(df)
                    
                    # Calculate BB if not in context
                    bb_upper = htf.get("bb_upper_5m", 0.0)
                    bb_lower = htf.get("bb_lower_5m", 0.0)
                    
                    if bb_upper == 0.0 or bb_lower == 0.0:
                        from analysis_module.technical import TechnicalAnalyzer
                        analyzer = TechnicalAnalyzer("TEMP")
                        bb_data = analyzer._calculate_bollinger_bands(df)
                        if bb_data and 'upper' in bb_data:
                            bb_upper = bb_data['upper'].iloc[-1]
                            bb_lower = bb_data['lower'].iloc[-1]
                    
                    # Get RSI
                    rsi_5 = htf.get("rsi_5", 50)
                    
                    # Build technical context for combo
                    technical_context = {
                        "macd": macd_data,
                        "rsi_5": rsi_5,
                        "bb_upper": bb_upper,
                        "bb_lower": bb_lower
                    }
                    
                    #Evaluate combo signal
                    combo_result = self.combo_evaluator.evaluate_signal(
                        df=df,
                        direction_bias=direction,
                        technical_context=technical_context
                    )
                    
                    # Apply combo scoring
                    combo_strength = combo_result['strength']
                    combo_score = combo_result['score']
                    
                    if combo_strength == 'STRONG':
                        score += 15
                        reasons.append(f"🔥 STRONG Combo ({combo_score}/3: {combo_result['details']}) (+15)")
                    elif combo_strength == 'MEDIUM':
                        score += 10
                        reasons.append(f"✅ MEDIUM Combo ({combo_score}/3: {combo_result['details']}) (+10)")
                    elif combo_strength == 'WEAK':
                        score += 0
                        reasons.append(f"⚠️ WEAK Combo ({combo_score}/3: {combo_result['details']}) (+0)")
                    else:  # INVALID
                        score -= 10
                        reasons.append(f"❌ INVALID Combo ({combo_result['details']}) (-10)")
                    
                    # Store combo result in signal data for logging/alerts
                    sig_data['combo_signal'] = combo_result
                    
            except Exception as e:
                logger.warning(f"Combo signal evaluation failed: {e}")

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

```

### ./analysis_module/technical.py

```python
"""
Technical Analysis Module
Core calculations: PDH/PDL, Support/Resistance, Volume, Breakouts, Advanced TA
Includes debugging output for signal validation
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from config.settings import (
    MAX_SAME_DIRECTION_ALERTS,
    MIN_SIGNAL_CONFIDENCE,
    LOOKBACK_BARS,
    MIN_SR_TOUCHES,
    VOLUME_PERIOD,
    MIN_VOLUME_RATIO,
    SR_CLUSTER_TOLERANCE,
    BREAKOUT_CONFIRMATION_CANDLES,
    FALSE_BREAKOUT_RETRACEMENT,
    RETEST_ZONE_PERCENT,
    MIN_RSI_BULLISH,
    MAX_RSI_BEARISH,
    RSI_PERIOD,
    ATR_PERIOD,
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER,
    EMA_SHORT,
    EMA_LONG,
    get_tick_size,  # NEW: for correct tick size calculation
    MIN_RISK_REWARD_RATIO,
    DEBUG_MODE,
)

from analysis_module.adaptive_thresholds import AdaptiveThresholds

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
    BULLISH_PIN_BAR = "BULLISH_PIN_BAR"
    BEARISH_PIN_BAR = "BEARISH_PIN_BAR"
    BULLISH_ENGULFING = "BULLISH_ENGULFING"
    BEARISH_ENGULFING = "BEARISH_ENGULFING"


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
    atr: float = 0.0
    debug_info: Dict = None
    take_profit_2: float = 0.0
    take_profit_3: float = 0.0


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
    rsi_divergence: str = "NONE"
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    ema_50: float = 0.0
    # Fibonacci Pivot Levels
    r1_fib: float = 0.0
    s1_fib: float = 0.0
    r2_fib: float = 0.0
    s2_fib: float = 0.0
    # Confluence zones
    confluence_zones: List[Dict] = None


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
            
            # Add PDH/PDL to levels before clustering (Crucial for Breakout detection)
            if pdh: resistance_levels.append(pdh)
            if pdl: support_levels.append(pdl)
            
            # Calculate daily pivots from PDH/PDL/PDC
            pivots = {}
            if pdh and pdl:
                try:
                    # In a real setup, calculate_pdh_pdl should also return pdc
                    # Using resample to find previous day close
                    daily_data = df.resample("D").agg({"close": "last"}).dropna()
                    if len(daily_data) >= 2:
                        pdc = daily_data.iloc[-2]["close"]
                        pivots = self._calculate_pivots(pdh, pdl, pdc)
                        
                        # Add Fibonacci Pivots to structured levels
                        if "s1_fib" in pivots: support_levels.append(pivots["s1_fib"])
                        if "r1_fib" in pivots: resistance_levels.append(pivots["r1_fib"])
                        if "s2_fib" in pivots: support_levels.append(pivots["s2_fib"])
                        if "r2_fib" in pivots: resistance_levels.append(pivots["r2_fib"])
                except Exception as e:
                    logger.warning(f"⚠️ Pivot PDC fallback: {e}")
            
            pivot_std = pivots.get("pivot", 0.0)

            atr = self._calculate_atr(df_sub)
            volatility_score = self._calculate_volatility_score(df_sub)

            msg = (
                f"✅ S/R Calculated | Supports: {len(support_clusters)} {support_clusters[:3]}... | "
                f"Resistances: {len(resistance_clusters)} {resistance_clusters[:3]}..."
            )
            logger.info(msg)

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
                r1_fib=pivots.get("r1_fib", 0.0),
                s1_fib=pivots.get("s1_fib", 0.0),
                r2_fib=pivots.get("r2_fib", 0.0),
                s2_fib=pivots.get("s2_fib", 0.0),
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

    def _calculate_multi_targets(
        self,
        entry_price: float,
        stop_loss: float,
        direction: str,
        support_resistance: TechnicalLevels,
        is_strong_trend: bool = False
    ) -> Tuple[float, float, float]:
        """
        Calculate T1, T2, T3 based on S/R levels and R:R.
        
        T1 (Safe): Maximum of (Nearest Level, 1:1.5 RR if level too close)
        T2 (Moderate): Next Key Level or 1:2 RR (Only if strong trend)
        T3 (Aggressive): Level after that or 1:3 RR (Only if strong trend)
        
        Returns: (t1, t2, t3)
        """
        atr = support_resistance.atr
        risk = abs(entry_price - stop_loss)
        if risk == 0:
            return 0.0, 0.0, 0.0
            
        t1, t2, t3 = 0.0, 0.0, 0.0
        
        if direction.upper() == "LONG":
            # Identify resistance levels above entry
            relevant_levels = [r for r in sorted(support_resistance.resistance_levels) if r > entry_price]
            
            # --- T1 ---
            if relevant_levels:
                t1 = relevant_levels[0]
                # If risk reward < 1:1, push to next level or forcing 1.5R
                if (t1 - entry_price) < risk:
                     t1 = max(t1, entry_price + (risk * 1.5))
            else:
                t1 = entry_price + (risk * 1.5)
                
            # If trend is weak, stop here
            if not is_strong_trend:
                # Round to tick size before returning
                tick_size = get_tick_size(self.instrument, is_option=False)
                t1 = round(t1 / tick_size) * tick_size
                return t1, 0.0, 0.0
                
            # --- T2 ---
            if len(relevant_levels) > 1:
                t2 = relevant_levels[1]
            else:
                t2 = entry_price + (risk * 2.5)
            if t2 <= t1:
                t2 = max(t2, t1 + (risk * 1.0))

            # --- T3 ---
            # Cap at 3x for realistic intraday targets (was 4x)
            if len(relevant_levels) > 2:
                t3 = relevant_levels[2]
            else:
                t3 = entry_price + (risk * 3.0)  # Changed from 4.0
            if t3 <= t2:
                t3 = max(t3, t2 + (risk * 1.0))
                
        else: # SHORT
            # Identify support levels below entry (High -> Low for iteration)
            relevant_levels = [s for s in sorted(support_resistance.support_levels, reverse=True) if s < entry_price]
            
            # --- T1 ---
            if relevant_levels:
                t1 = relevant_levels[0]
                if (entry_price - t1) < risk:
                    t1 = min(t1, entry_price - (risk * 1.5))
            else:
                t1 = entry_price - (risk * 1.5)
                
            # If trend is weak, stop here
            if not is_strong_trend:
                # Round to tick size before returning
                tick_size = get_tick_size(self.instrument, is_option=False)
                t1 = round(t1 / tick_size) * tick_size
                return t1, 0.0, 0.0
                
            # --- T2 ---
            if len(relevant_levels) > 1:
                t2 = relevant_levels[1]
            else:
                t2 = entry_price - (risk * 2.5)
            if t2 >= t1:
                 t2 = min(t2, t1 - (risk * 1.0))
                 
            # --- T3 ---
            # Cap at 3x for realistic intraday targets (was 4x)
            if len(relevant_levels) > 2:
                t3 = relevant_levels[2]
            else:
                t3 = entry_price - (risk * 3.0)  # ← Changed from 4.0 to 3.0
            if t3 >= t2:
                t3 = min(t3, t2 - (risk * 1.0))
        
        # CRITICAL: Round all targets to correct tick size
        # NIFTY spot uses 1.0, NOT 0.05
        tick_size = get_tick_size(self.instrument, is_option=False)
        t1 = round(t1 / tick_size) * tick_size
        t2 = round(t2 / tick_size) * tick_size if t2 > 0 else 0.0
        t3 = round(t3 / tick_size) * tick_size if t3 > 0 else 0.0
                
        return t1, t2, t3

    def _calculate_pivots(
        self, high: float, low: float, close: float
    ) -> Dict[str, float]:
        """
        Calculate pivot points (Standard & Fibonacci-style).
        Calculated from Previous Day High, Low, and Close.
        """
        try:
            pivot = (high + low + close) / 3
            range_diff = high - low

            # Standard Pivots
            r1_std = (2 * pivot) - low
            s1_std = (2 * pivot) - high
            r2_std = pivot + range_diff
            s2_std = pivot - range_diff

            # Fibonacci Pivots (Commonly used by professionals)
            r1_fib = pivot + (0.382 * range_diff)
            s1_fib = pivot - (0.382 * range_diff)
            r2_fib = pivot + (0.618 * range_diff)
            s2_fib = pivot - (0.618 * range_diff)
            r3_fib = pivot + (1.000 * range_diff)
            s3_fib = pivot - (1.000 * range_diff)

            return {
                "pivot": pivot,
                "r1": r1_std,
                "s1": s1_std,
                "r2": r2_std,
                "s2": s2_std,
                "r1_fib": r1_fib,
                "s1_fib": s1_fib,
                "r2_fib": r2_fib,
                "s2_fib": s2_fib,
                "r3_fib": r3_fib,
                "s3_fib": s3_fib
            }
        except Exception as e:
            logger.error(f"Error calculating pivots: {e}")
            return {}

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands (Upper, Middle, Lower)."""
        try:
            middle = df['close'].rolling(window=period).mean()
            std = df['close'].rolling(window=period).std()
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)
            
            return {
                "upper": upper,
                "middle": middle,
                "lower": lower
            }
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {}

    def _detect_rsi_divergence(self, df: pd.DataFrame, lookback: int = 10) -> str:
        """
        Detect Bullish or Bearish RSI Divergence over the last 'lookback' candles.
        
        Returns: 'BULLISH', 'BEARISH', or 'NONE'
        """
        try:
            if len(df) < lookback + 5 or "rsi" not in df.columns:
                return "NONE"
            
            recent = df.tail(lookback)
            
            # Find recent price pivots
            price_highs = df['high'].rolling(3, center=True).apply(lambda x: x[1] == max(x), raw=True).astype(bool)
            price_lows = df['low'].rolling(3, center=True).apply(lambda x: x[1] == min(x), raw=True).astype(bool)
            
            # BEARISH DIVERGENCE: Price Higher High, RSI Lower High
            # Check for two recent high peaks
            high_indices = df[price_highs].index[-2:] if any(price_highs) else []
            if len(high_indices) == 2:
                p1, p2 = high_indices
                if df.loc[p2, 'high'] >= df.loc[p1, 'high'] and df.loc[p2, 'rsi'] < df.loc[p1, 'rsi']:
                    return "BEARISH"
                # Variation: Price tests same resistance but RSI is lower (as in expert report)
                if abs(df.loc[p2, 'high'] - df.loc[p1, 'high']) < (df.loc[p1, 'high'] * 0.0005) and df.loc[p2, 'rsi'] < df.loc[p1, 'rsi'] - 2:
                    return "BEARISH"

            # BULLISH DIVERGENCE: Price Lower Low, RSI Higher Low
            low_indices = df[price_lows].index[-2:] if any(price_lows) else []
            if len(low_indices) == 2:
                l1, l2 = low_indices
                if df.loc[l2, 'low'] <= df.loc[l1, 'low'] and df.loc[l2, 'rsi'] > df.loc[l1, 'rsi']:
                    return "BULLISH"
                    
            return "NONE"
        except Exception as e:
            logger.error(f"Error detecting RSI divergence: {e}")
            return "NONE"

    def validate_level_interaction(
        self,
        price: float,
        level: float,
        direction: str,
        tolerance: float = 3.0  # ±3 points for Nifty at ~25,000
    ) -> tuple[bool, float]:
        """
        Check if price is interacting with level precisely (within ±tolerance).
        
        Expert's methodology: Only enter when price is within ±3 points of key level.
        
        Args:
            price: Current price
            level: Key level to check (PDH, S1, R1, etc.)
            direction: "SHORT" or "LONG"
            tolerance: Maximum distance in points (default 3.0)
            
        Returns:
            (is_valid, distance) - True if price is within tolerance, actual distance
        """
        distance = abs(price - level)
        
        # Check if within tolerance
        if distance > tolerance:
            return False, distance
            
        # For SHORT: price should be near or above level (testing resistance)
        # For LONG: price should be near or below level (testing support)
        if direction == "SHORT":
            # Allow entry from 0-3 points above resistance
            is_valid = price >= level - tolerance and price <= level + tolerance
        else:
            # Allow entry from 0-3 points below support
            is_valid = price >= level - tolerance and price <= level + tolerance
            
        return is_valid, distance

    def calculate_fibonacci_levels(self, df: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        """
        Calculate Fibonacci Retracement levels from recent Swing High/Low.
        
        Args:
            df: OHLCV DataFrame
            lookback: Number of candles to scan for Swing High/Low (default 50)
            
        Returns:
            Dict with '0.382', '0.5', '0.618' levels
        """
        try:
            recent_data = df.tail(lookback)
            if recent_data.empty:
                return {}
            
            # Find Swing High and Low
            swing_high = recent_data["high"].max()
            swing_low = recent_data["low"].min()
            range_diff = swing_high - swing_low
            
            # Determine direction (Trend) to know if we represent support or resistance
            # For simplicity in this context, we return the Retracement levels from the Low upwards 
            # (Support in uptrend) AND High downwards (Resistance in downtrend) is too complex for now.
            # We will calculate "Retracement from Low to High" (assuming uptrend support checks)
            # and "Retracement from High to Low" (assuming downtrend resistance checks).
            
            # Just returning plain levels relative to the range is ambiguous without trend.
            # Standard approach: Return both "Retracement Up" (Support) and "Retracement Down" (Resistance) keys?
            # Or simplified: Just levels.
            
            # Let's assume we are looking for SUPPORT levels (pullback in uptrend)
            # 0% is High, 100% is Low.
            # 38.2% Retracement = High - (Range * 0.382)
            # 50% Retracement = High - (Range * 0.5)
            # 61.8% Retracement = High - (Range * 0.618)
            
            fib_support = {
                "0.236": swing_high - (range_diff * 0.236),
                "0.382": swing_high - (range_diff * 0.382),
                "0.5": swing_high - (range_diff * 0.5),
                "0.618": swing_high - (range_diff * 0.618),
                "0.786": swing_high - (range_diff * 0.786),
                "swing_high": swing_high,
                "swing_low": swing_low
            }
            
            return fib_support
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return {}

    # =====================================================================
    # VOLUME ANALYSIS
    # =====================================================================

    def check_volume_confirmation(
        self, df: pd.DataFrame
    ) -> Tuple[bool, float, str]:
        """
        Check if current candle has volume confirmation.
        Returns True if volume is zero (index data) to avoid blocking signals.
        """
        try:
            df = df.copy()
            if len(df) < VOLUME_PERIOD + 2:
                return False, 0.0, ""

            # Check if volume data exists (indices often have 0 volume)
            total_volume = df["volume"].sum()
            if total_volume == 0:
                logger.debug("⚠️  Zero volume detected (Index data) - Bypassing volume check")
                return True, 1.0, "Index (No Vol)"

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
            # Default to True on error to not block price signals if data is bad
            return True, 0.0, "Error (Bypassed)"

    def _calculate_rvol(self, df: pd.DataFrame) -> Tuple[float, str]:
        """
        Calculate RVOL (Relative Volume) - current volume vs average for this time of day.
        
        More accurate than simple volume ratio as it accounts for time-of-day patterns
        (e.g., higher volume at market open/close).
        
        Returns:
            (rvol, description)
        """
        try:
            if len(df) < 20:
                return 1.0, "Insufficient data"
            
            # Check if volume data exists
            total_volume = df["volume"].sum()
            if total_volume == 0:
                return 1.0, "Index (No Vol)"
            
            current_vol = float(df["volume"].iloc[-1])
            
            # Simple RVOL: current vs recent 20-period average
            # (For true time-of-day RVOL, would need multi-day data with timestamps)
            avg_vol = float(df["volume"].tail(20).mean())
            
            rvol = current_vol / avg_vol if avg_vol > 0 else 1.0
            
            description = f"RVOL {rvol:.2f}x (Vol: {current_vol:.0f} vs Avg: {avg_vol:.0f})"
            
            return rvol, description
        
        except Exception as e:
            logger.warning(f"⚠️ RVOL calculation failed: {e}")
            return 1.0, "RVOL calc error"

    # =====================================================================
    # OPENING RANGE BREAKOUT (ORB)
    # =====================================================================

    def get_opening_range(self, df: pd.DataFrame, duration_mins: int = 15) -> Optional[Dict]:
        """
        Calculate opening range (first N-minute candle of trading session).
        
        ORB is one of the most reliable intraday setups for Nifty/BankNifty.
        Breakouts of the opening range tend to continue in that direction.
        
        Args:
            df: Intraday OHLC data with datetime index (5m, 15m, or 30m)
            duration_mins: 15 or 30 minutes (default 15)
        
        Returns:
            Dict with ORB high/low/range or None if insufficient data
        """
        try:
            # Market opens at 9:15 AM IST
            # We want the FIRST candle of the specified duration, not all candles in the range
            
            # For 15m candle: Take the candle at 09:15 (covers 09:15-09:30)
            # For 30m candle: Take candles from 09:15-09:45
            
            # Filter to get candles for the LATEST day in the dataset only
            # This prevents picking the opening candle from 5 days ago in a 5d dataset
            latest_date = df.index.max().date()
            df_today = df[df.index.date == latest_date]
            
            market_open_candles = df_today[df_today.index.time >= time(9, 15)]
            
            if market_open_candles.empty:
                logger.debug(f"⏭️ No data from market open time (09:15) on {latest_date}")
                return None
            
            # Get the first candle(s) based on duration
            if duration_mins == 15:
                # For 15m: Just take the first candle (09:15-09:30)
                opening_candles = market_open_candles.iloc[:1]
            elif duration_mins == 30:
                # For 30m: Take first 2 candles if using 15m data, or first candle if using 30m data
                # We can detect interval from index
                if len(market_open_candles) >= 2:
                    diff = (market_open_candles.index[1] - market_open_candles.index[0]).total_seconds() / 60
                    num_candles = duration_mins // int(diff) if diff > 0 else 1
                else:
                    num_candles = 1
                opening_candles = market_open_candles.iloc[:num_candles]
            else:
                # Default to 15m
                opening_candles = market_open_candles.iloc[:1]
            
            if opening_candles.empty:
                return None
            
            orb_high = float(opening_candles["high"].max())
            orb_low = float(opening_candles["low"].min())
            orb_range = orb_high - orb_low
            
            logger.info(
                f"📊 Opening Range ({duration_mins}min) | "
                f"High: {orb_high:.2f} | Low: {orb_low:.2f} | Range: {orb_range:.2f}"
            )
            
            return {
                "high": orb_high,
                "low": orb_low,
                "range": orb_range,
                "duration_mins": duration_mins
            }
        
        except Exception as e:
            logger.error(f"❌ Opening range calculation failed: {e}")
            return None

    # =====================================================================
    # HIGHER TF CONTEXT (15m)
    # =====================================================================

    def get_higher_tf_context(
        self,
        df_15m: pd.DataFrame,
        df_5m: pd.DataFrame = None,
        df_daily: pd.DataFrame = None,
        india_vix: float = None,
    ) -> Dict:
        """
        Build higher timeframe context:
        - 15m Trend direction: UP / DOWN / FLAT
        - 15m RSI and EMAs
        - 5m VWAP and 20 EMA (for intraday setups)
        - Previous day trend
        """
        context = {
            "trend_direction": "FLAT",
            "ema_short_15": 0.0,
            "ema_long_15": 0.0,
            "rsi_15": 50.0,
            "vwap_5m": 0.0,
            "vwap_slope": "FLAT",
            "ema_20_5m": 0.0,
            "price_above_vwap": False,
            "price_above_ema20": False,
            "prev_day_trend": "FLAT",
        }

        try:
            # ====================
            # 15m Trend Analysis
            # ====================
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

            if ema_short_15 > ema_long_15:
                trend = "UP"
            elif ema_short_15 < ema_long_15:
                trend = "DOWN"
            else:
                trend = "FLAT"
            
            # ====================
            # Early Reversal Detection (reduce lag)
            # ====================
            # If trend is DOWN but price > 20EMA and RSI > 55 -> weak UP (Early Reversal)
            current_close = float(df["close"].iloc[-1])
            ema_20_15m = float(df["close"].ewm(span=20, adjust=False).mean().iloc[-1])
            
            if trend == "DOWN" and current_close > ema_20_15m and rsi_15 > 55:
                trend = "UP"  # Early reversal
                logger.info(f"🔄 Early trend reversal detected (Price > 15m EMA20 & RSI {rsi_15:.1f})")
            
            elif trend == "UP" and current_close < ema_20_15m and rsi_15 < 30:  # More conservative (was 45)
                trend = "DOWN" # Early correction
                logger.info(f"🔄 Early trend correction detected (Price < 15m EMA20 & RSI {rsi_15:.1f})")

            context.update(
                {
                    "trend_direction": trend,
                    "ema_short_15": ema_short_15,
                    "ema_long_15": ema_long_15,
                    "rsi_15": rsi_15,
                    "ema_50_15m": float(df["close"].ewm(span=50, adjust=False).mean().iloc[-1]),
                    "rsi_divergence_15m": self._detect_rsi_divergence(df, lookback=15),
                }
            )

            # ====================
            # 5m VWAP and 20 EMA
            # ====================
            if df_5m is not None and not df_5m.empty and len(df_5m) >= 20:
                df_5m_copy = df_5m.copy().sort_index()
                
                # Calculate VWAP
                _, vwap_5m, vwap_slope = self._calculate_vwap(df_5m_copy)
                
                # Calculate 20 EMA on 5m
                ema_20 = df_5m_copy["close"].ewm(span=20, adjust=False).mean()
                ema_20_5m = float(ema_20.iloc[-1])
                ema_50_5m = float(df_5m_copy["close"].ewm(span=50, adjust=False).mean().iloc[-1])
                
                # Bollinger Bands
                bb = self._calculate_bollinger_bands(df_5m_copy)
                bb_upper = float(bb["upper"].iloc[-1])
                bb_lower = float(bb["lower"].iloc[-1])

                # RSI Divergence on 5m
                rsi_5m_series = self._calculate_rsi_series(df_5m_copy)
                df_5m_copy["rsi"] = rsi_5m_series
                rsi_div_5m = self._detect_rsi_divergence(df_5m_copy)
                
                # Current price vs VWAP and EMA
                current_price = float(df_5m_copy["close"].iloc[-1])
                price_above_vwap = current_price > vwap_5m
                price_above_ema20 = current_price > ema_20_5m
                
                context.update({
                    "vwap_5m": vwap_5m,
                    "vwap_slope": vwap_slope,
                    "ema_20_5m": ema_20_5m,
                    "ema_50_5m": ema_50_5m,
                    "bb_upper_5m": bb_upper,
                    "bb_lower_5m": bb_lower,
                    "rsi_divergence_5m": rsi_div_5m,
                    "price_above_vwap": price_above_vwap,
                    "price_above_ema20": price_above_ema20,
                })
            
            # ====================
            # Adaptive RSI Thresholds (Phase 2 + Phase 3)
            # ====================
            # Ensure df_5m_copy is available even if 20-EMA block was skipped
            if df_5m is not None and not df_5m.empty:
                df_5m_copy = df_5m.copy().sort_index()
                
                # Ensure ATR is calculated (needed for adaptive thresholds)
                if "atr" not in df_5m_copy.columns:
                    high_low = df_5m_copy["high"] - df_5m_copy["low"]
                    high_close = (df_5m_copy["high"] - df_5m_copy["close"].shift()).abs()
                    low_close = (df_5m_copy["low"] - df_5m_copy["close"].shift()).abs()
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    df_5m_copy["atr"] = ranges.max(axis=1).rolling(14).mean()

                # Calculate ATR percentile and get dynamic RSI thresholds
                if "atr" in df_5m_copy.columns and len(df_5m_copy) >= 60:
                    current_atr = float(df_5m_copy["atr"].iloc[-1])
                    atr_percentile = AdaptiveThresholds.calculate_atr_percentile(
                        df_5m_copy, current_atr, lookback=60
                    )
                    
                    # NEW: Use the india_vix passed as argument
                    # Use VIX if available, otherwise ATR percentile
                    rsi_long_threshold, rsi_short_threshold = AdaptiveThresholds.get_rsi_thresholds(
                        vix=india_vix,  # Will use this if not None
                        atr_percentile=atr_percentile  # Fallback
                    )
                    
                    context.update({
                        "atr_percentile": atr_percentile,
                        "india_vix": india_vix,  # Store for reference
                        "rsi_long_threshold": rsi_long_threshold,
                        "rsi_short_threshold": rsi_short_threshold,
                    })
                
                # Enhanced logging with VIX info
                if india_vix:
                    logger.info(
                        f"📊 Adaptive RSI (VIX): VIX {india_vix:.1f} → RSI {rsi_short_threshold}/{rsi_long_threshold}"
                    )
                else:
                    logger.info(
                        f"📊 Adaptive RSI (ATR): ATR %ile {atr_percentile:.1f} → RSI {rsi_short_threshold}/{rsi_long_threshold}"
                    )

            # ====================
            # Previous Day Trend
            # ====================
            if df_daily is not None:
                prev_day_trend = self._get_previous_day_trend(df_daily)
                context["prev_day_trend"] = prev_day_trend

            # ====================
            # Opening Range (ORB)
            # ====================
            if df_5m is not None:
                orb = self.get_opening_range(df_5m, duration_mins=15)
                if orb:
                    context["opening_range"] = orb
                    context["orb_high"] = orb["high"]
                    context["orb_low"] = orb["low"]
                    context["orb_range"] = orb["range"]

            logger.info(
                "📐 Higher TF context | "
                f"Trend15m: {trend} | "
                f"VWAP: {context.get('vwap_5m', 0):.2f} ({context.get('vwap_slope', 'N/A')}) | "
                f"20EMA: {context.get('ema_20_5m', 0):.2f} | "
                f"PrevDay: {context.get('prev_day_trend', 'N/A')}"
            )
            return context

        except Exception as e:
            logger.error(f"❌ get_higher_tf_context failed: {str(e)}")
            return context

    # =====================================================================
    # BREAKOUT QUALITY FILTERS
    # =====================================================================

    def _detect_consolidation(
        self, df: pd.DataFrame, lookback: int = 20
    ) -> Optional[Dict]:
        """
        Detect if price is consolidating (sideways range).
        
        High-quality breakouts come from consolidation, not random noise.
        
        Args:
            df: OHLCV DataFrame
            lookback: Bars to analyze
            
        Returns:
            {
                "is_consolidating": bool,
                "range_high": float,
                "range_low": float,
                "range_pct": float,
                "bars_in_range": int
            }
        """
        try:
            if df is None or len(df) < lookback:
                return None
            
            recent = df.tail(lookback)
            
            # Calculate range
            range_high = float(recent["high"].max())
            range_low = float(recent["low"].min())
            range_pct = ((range_high - range_low) / range_low) * 100.0
            
            # Consolidation criteria:
            # 1. Tight range (< 2% for intraday on indices)
            # 2. Majority of bars within this range (at least 70%)
            # 3. Minimum bars in consolidation (at least 8)
            
            is_tight = range_pct < 2.0
            
            # Count bars fully within range
            bars_in_range = sum(
                1 for _, row in recent.iterrows()
                if range_low <= row["low"] and row["high"] <= range_high
            )
            
            pct_in_range = bars_in_range / lookback
            is_consolidated = is_tight and pct_in_range >= 0.7 and bars_in_range >= 8
            
            result = {
                "is_consolidating": is_consolidated,
                "range_high": range_high,
                "range_low": range_low,
                "range_pct": range_pct,
                "bars_in_range": bars_in_range,
            }
            
            if is_consolidated:
                logger.info(
                    f"📦 Consolidation detected | Range: {range_low:.2f}-{range_high:.2f} "
                    f"({range_pct:.2f}%) | {bars_in_range}/{lookback} bars"
                )
            else:
                logger.debug(
                    f"⏭️ No consolidation | Range: {range_pct:.2f}% | "
                    f"Bars in range: {bars_in_range}/{lookback}"
                )
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Consolidation detection failed: {str(e)}")
            return None

    def _detect_volume_surge(
        self, df: pd.DataFrame
    ) -> Tuple[bool, float, str]:
        """
        Detect volume surge on current bar.
        
        Strong breakouts have volume surges, not just average volume.
        
        Criteria:
            - Current volume > 1.5x 20-bar average
            - AND Current volume > highest of last 5 bars
            
        Returns:
            (has_surge, surge_ratio, description)
        """
        try:
            if df is None or len(df) < 20:
                return False, 0.0, "Insufficient data"
            
            # Check if volume data exists
            if df["volume"].sum() == 0:
                logger.debug("⚠️ No volume data - Treating as NO SURGE (Require Consolidation)")
                return False, 0.0, "Index (No Vol)"
            
            current_vol = float(df["volume"].iloc[-1])
            avg_vol_20 = float(df["volume"].tail(20).mean())
            max_vol_5 = float(df["volume"].tail(6).iloc[:-1].max())  # Exclude current
            
            surge_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 0
            
            # Surge conditions
            # Restored to use configured MIN_VOLUME_RATIO (default 1.5) as 1.2 was too lenient
            # causing too many false breakout signals
            above_average = surge_ratio >= MIN_VOLUME_RATIO
            above_recent = current_vol > max_vol_5
            
            has_surge = above_average and above_recent
            
            if has_surge:
                description = f"Volume surge: {surge_ratio:.2f}x avg, highest in 5 bars"
                logger.info(f"📊 {description}")
            else:
                description = f"No surge: {surge_ratio:.2f}x avg (need {MIN_VOLUME_RATIO}x + highest in 5)"
                logger.debug(f"⏭️ {description}")
            
            return has_surge, surge_ratio, description
            
        except Exception as e:
            logger.warning(f"⚠️ Volume surge detection failed: {str(e)}")
            # Default to allowing the trade if check fails
            return True, 1.0, "Error (bypassed)"

    def _is_valid_breakout_time(self) -> Tuple[bool, str]:
        """
        Check if current time is suitable for breakout trading.
        
        Avoid:
            - First 15 mins (09:15-09:30): Too volatile, whipsaws
            - Last hour (14:30-15:30): Low follow-through
            - Lunch hour (12:30-13:30): Low volume, choppy
            
        Best breakout hours: 09:30-12:30, 13:30-14:30
        
        Returns:
            (is_valid, reason)
        """
        try:
            import pytz
            
            ist = pytz.timezone("Asia/Kolkata")
            now = datetime.now(ist).time()
            
            # Morning volatility window
            if time(9, 15) <= now < time(9, 20):
                return False, "Morning volatility (09:15-09:20)"
            
            # Lunch hour (low volume)
            if time(12, 30) <= now < time(13, 0):
                return False, "Lunch hour (12:30-13:00)"
            
            # Last hour (poor follow-through)
            if time(14, 30) <= now <= time(15, 30):
                return False, "Last hour (14:30-15:30)"
            
            # Good breakout windows
            # Relaxed start to 09:20 to catch early trends
            # Europe open approach at 13:00
            if (time(9, 20) <= now < time(12, 30)) or \
               (time(13, 0) <= now < time(14, 30)):
                return True, "Optimal breakout window"
            
            return False, "Outside breakout hours"
            
        except Exception as e:
            logger.warning(f"⚠️ Time check failed: {str(e)}")
            return True, "Time check bypassed"

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
        Detect high-quality breakout with multi-factor confirmation.
        
        Filters:
        1. Time of day (avoid choppy periods)
        2. Consolidation (must break from range)
        3. MTF Trend (15m alignment)
        4. Volume Surge (explosive move)
        5. RSI Momentum
        """
        try:
            if df is None or df.empty:
                return None

            # 1. Time of Day Filter
            is_valid_time, time_reason = self._is_valid_breakout_time()
            if not is_valid_time:
                # Log only occasionally to avoid spam
                if datetime.now().minute % 15 == 0:
                    logger.debug(f"⏭️ Breakout skipped: {time_reason}")
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
            
            # 2. Volume Surge Check
            has_surge, surge_ratio, surge_desc = self._detect_volume_surge(df)
            
            # 3. Consolidation Check
            # We check this only if we find a potential breakout level
            consolidation = self._detect_consolidation(df)
            is_consolidating = consolidation["is_consolidating"] if consolidation else False

            # 4. Opening Range Breakout (ORB) Check
            orb = higher_tf_context.get("opening_range")
            is_orb_breakout = False
            orb_direction = None
            orb_boost = 0
            
            if orb:
                current_time = df.index[-1].time()
                # Only check ORB after it's established (after 9:30 AM)
                if current_time > time(9, 30):
                    # Bullish ORB: price breaks above opening range high
                    if current_high > orb["high"] * 1.0005:  # 0.05% buffer
                        is_orb_breakout = True
                        orb_direction = "BULLISH"
                        orb_boost = 10  # Boost confidence for ORB breakouts
                        logger.info(
                            f"🚀 ORB BULLISH BREAKOUT | Price: {current_price:.2f} > "
                            f"ORB High: {orb['high']:.2f}"
                        )
                    
                    # Bearish ORB: price breaks below opening range low
                    elif current_low < orb["low"] * 0.9995:  # 0.05% buffer
                        is_orb_breakout = True
                        orb_direction = "BEARISH"
                        orb_boost = 10  # Boost confidence for ORB breakouts
                        logger.info(
                            f"📉 ORB BEARISH BREAKOUT | Price: {current_price:.2f} < "
                            f"ORB Low: {orb['low']:.2f}"
                        )

            # ------------------------
            # Bullish breakout
            # ------------------------

                    # Standard Breakout Check
                    is_standard_breakout = False
                    nearest_resistance = 0.0
                    
                    if support_resistance.resistance_levels:
                         # Robustly find nearest resistance
                        sorted_resistances = sorted(
                            support_resistance.resistance_levels, 
                            key=lambda x: abs(x - current_price)
                        )
                        nearest_resistance = float(sorted_resistances[0])
                        if current_high > nearest_resistance:
                            is_standard_breakout = True
                    
                    # Combine triggers
                    # If ORB breakout, we force the level to be ORB High
                    if is_orb_breakout and orb_direction == "BULLISH":
                         breakout_level = orb["high"]
                         breakout_type = "ORB"
                    elif is_standard_breakout:
                         breakout_level = nearest_resistance
                         breakout_type = "S/R"
                    else:
                         breakout_level = 0.0
                         breakout_type = None

                    if breakout_type:
                        
                        # Filter: Must be consolidating OR have massive volume surge
                        # EXCEPTION: ORB and PDH breaks don't always need consolidation
                        is_major_level = (breakout_type == "ORB") or (abs(breakout_level - support_resistance.pdh) < 1.0)
                        
                        # NEW: For indices, relax consolidation requirement if strong trend
                        is_index = "NIFTY" in self.instrument.upper() or "INDEX" in self.instrument.upper()
                        strong_trend = (trend_dir == "UP" and rsi_15 >= 50)
                        
                        if is_index and strong_trend:
                            # Allow signals in strong trends even without consolidation
                            logger.info(f"✅ Index strong trend override (RSI {rsi_15:.1f}) - Consolidation not required")
                        elif not is_consolidating and not has_surge and not is_major_level:
                            logger.info(
                                f"⏭️ Bullish breakout ignored (No consolidation/surge) | "
                                f"Vol: {surge_ratio:.1f}x"
                            )
                            return None
                        
                        # NEW: RVOL Filter (Relative Volume check)
                        # Skip RVOL check for indices (they have no volume data)
                        is_index = "NIFTY" in self.instrument.upper() or "INDEX" in self.instrument.upper()
                        
                        if not is_index:
                            rvol, rvol_desc = self._calculate_rvol(df)
                            MIN_RVOL_BREAKOUT = 1.5  # Require 1.5x relative volume for breakouts
                            
                            if rvol < MIN_RVOL_BREAKOUT and not is_major_level:
                                logger.info(
                                    f"⏭️ Bullish breakout ignored (Low RVOL) | "
                                    f"{rvol_desc} < {MIN_RVOL_BREAKOUT}x required"
                                )
                                return None
                            
                            logger.info(f"✅ RVOL check passed | {rvol_desc}")
                        else:
                            logger.debug(f"⏭️ RVOL check skipped (index instrument)")
                            
                        # Filter: MTF Trend Alignment
                        # Use ADAPTIVE RSI thresholds from context (fallback to static)
                        rsi_long_threshold = higher_tf_context.get("rsi_long_threshold", MIN_RSI_BULLISH)
                        
                        mtf_ok = (
                            trend_dir in ["UP", "NEUTRAL"]
                            and rsi_15 >= rsi_long_threshold  # Adaptive threshold
                        )
                        
                        if is_major_level: mtf_ok = True

                        if not mtf_ok:
                            logger.info(
                                "⏭️  Bullish breakout ignored (MTF filter) | "
                                f"Trend15m: {trend_dir} | RSI15: {rsi_15:.1f}"
                            )
                        else:
                            logger.info(
                                f"🚀 Bullish ({breakout_type}) breakout candidate | "
                                f"Price: {current_price:.2f} > Lvl: {breakout_level:.2f} | "
                                f"Consolidation: {is_consolidating} | Surge: {has_surge}"
                            )

                            atr = support_resistance.atr
                            sl = current_low - (atr * ATR_SL_MULTIPLIER)
                            
                            # Strong Trend Logic
                            is_strong_trend = (
                                has_surge 
                                and (orb_direction == "BULLISH" or trend_dir in ["UP", "NEUTRAL"])
                                and rsi_15 >= rsi_long_threshold
                                and rsi_5 >= rsi_long_threshold
                            )
                            
                            tp1, tp2, tp3 = self._calculate_multi_targets(
                                current_price, sl, "LONG", support_resistance, is_strong_trend
                            )
                            
                            risk_reward = (tp1 - current_price) / max(
                                current_price - sl, 1e-6
                            )

                            # Confidence Scoring
                            confidence = 60.0
                            if has_surge:
                                confidence += 10
                            if is_consolidating:
                                confidence += 10
                            # RSI confirmation for 5m breakout
                            if rsi_5 >= rsi_long_threshold:  # Adaptive threshold
                                confidence += 5
                            if trend_dir == "UP":
                                confidence += 10
                            if risk_reward >= MIN_RISK_REWARD_RATIO:
                                confidence += 5
                            # NEW: ORB boost
                            if is_orb_breakout and orb_direction == "BULLISH":
                                confidence += orb_boost

                            confidence = min(confidence, 95.0)

                            breakout_signal = Signal(
                                signal_type=SignalType.BULLISH_BREAKOUT,
                                instrument=self.instrument,
                                timeframe="5MIN",
                                price_level=breakout_level,
                                entry_price=current_price,
                                stop_loss=sl,
                                take_profit=tp1,
                                take_profit_2=tp2,
                                take_profit_3=tp3,
                                confidence=confidence,
                                volume_confirmed=has_surge,
                                momentum_confirmed=(
                                    rsi_5 >= MIN_RSI_BULLISH
                                ),
                                risk_reward_ratio=risk_reward,
                                timestamp=pd.Timestamp.now(),
                                description=(
                                    f"Bullish breakout at {breakout_level:.2f} | "
                                    f"RR: {risk_reward:.2f} | "
                                    f"Consolidation: {is_consolidating}"
                                    f"{' | ORB Breakout' if is_orb_breakout and orb_direction == 'BULLISH' else ''}"
                                ),
                                debug_info={
                                    "surge_ratio": surge_ratio,
                                    "is_consolidating": is_consolidating,
                                    "rsi_5": rsi_5,
                                    "rsi_15": rsi_15,
                                    "trend_dir": trend_dir,
                                    "atr": atr,
                                    "is_strong_trend": is_strong_trend
                                },
                            )

            # ------------------------
            # Bearish breakdown
            # ------------------------

            # ------------------------
            # Bearish breakdown
            # ------------------------
            # Standard Breakdown Check
            is_standard_breakdown = False
            nearest_support = 0.0
            
            if support_resistance.support_levels:
                # Robustly find nearest support (closest to price)
                sorted_supports = sorted(
                    support_resistance.support_levels, 
                    key=lambda x: abs(x - current_price)
                )
                nearest_support = float(sorted_supports[0])
                if current_low < nearest_support:
                    is_standard_breakdown = True

            # Combine triggers
            if is_orb_breakout and orb_direction == "BEARISH":
                 breakdown_level = orb["low"]
                 breakdown_type = "ORB"
            elif is_standard_breakdown:
                 breakdown_level = nearest_support
                 breakdown_type = "S/R"
            else:
                 breakdown_level = 0.0
                 breakdown_type = None

            if breakdown_type:
                    
                # Exception: Major Levels (ORB/PDL)
                is_major_level = (breakdown_type == "ORB") or (support_resistance.pdl > 0 and abs(breakdown_level - support_resistance.pdl) < 1.0)

                # NEW: For indices, relax consolidation requirement if strong trend
                is_index = "NIFTY" in self.instrument.upper() or "INDEX" in self.instrument.upper()
                strong_trend = (trend_dir == "DOWN" and rsi_15 <= 50)
                
                if is_index and strong_trend:
                    # Allow signals in strong trends even without consolidation
                    logger.info(f"✅ Index strong trend override (RSI {rsi_15:.1f}) - Consolidation not required")
                elif not is_consolidating and not has_surge and not is_major_level:
                    logger.info(
                        f"⏭️ Bearish breakdown ignored (No consolidation/surge) | "
                        f"Vol: {surge_ratio:.1f}x"
                    )
                    return None
            
                # NEW: RVOL Filter (Relative Volume check)
                # Skip RVOL check for indices (they have no volume data)
                is_index = "NIFTY" in self.instrument.upper() or "INDEX" in self.instrument.upper()
                
                if not is_index:
                    rvol, rvol_desc = self._calculate_rvol(df)
                    MIN_RVOL_BREAKOUT = 1.5  # Require 1.5x relative volume
                    
                    if rvol < MIN_RVOL_BREAKOUT and not is_major_level:
                        logger.info(
                            f"⏭️ Bearish breakdown ignored (Low RVOL) | "
                            f"{rvol_desc} < {MIN_RVOL_BREAKOUT}x required"
                        )
                        return None
                    
                    logger.info(f"✅ RVOL check passed | {rvol_desc}")
                else:
                    logger.debug(f"⏭️ RVOL check skipped (index instrument)")
                 # Filter: MTF Alignment
                # Use ADAPTIVE RSI thresholds from context (fallback to static)
                rsi_short_threshold = higher_tf_context.get("rsi_short_threshold", MAX_RSI_BEARISH)
                
                mtf_ok = (
                    trend_dir in ["DOWN", "NEUTRAL"]
                    and rsi_15 <= rsi_short_threshold  # Adaptive threshold
                )
                
                if is_major_level: mtf_ok = True

                if not mtf_ok:
                    logger.info(
                        "⏭️  Bearish breakdown ignored (MTF filter) | "
                        f"Trend15m: {trend_dir} | RSI15: {rsi_15:.1f}"
                    )
                else:
                    logger.info(
                        f"📉 Bearish ({breakdown_type}) breakdown candidate | "
                        f"Price: {current_price:.2f} < Lvl: {breakdown_level:.2f} | "
                        f"Consolidation: {is_consolidating} | Surge: {has_surge}"
                    )

                    atr = support_resistance.atr
                    sl = current_high + (atr * ATR_SL_MULTIPLIER)
                    
                    # Strong Trend Logic
                    is_strong_trend = (
                        has_surge 
                        and trend_dir == "DOWN" 
                        and rsi_15 <= 40 
                        and rsi_5 <= 40
                    )
                    
                    # Calculate Multi-Targets
                    tp1, tp2, tp3 = self._calculate_multi_targets(
                        current_price, sl, "SHORT", support_resistance, is_strong_trend
                    )
                    
                    risk_reward = (current_price - tp1) / max(
                        sl - current_price, 1e-6
                    )

                    # Confidence Scoring
                    confidence = 60.0
                    if has_surge:
                        confidence += 10
                    if is_consolidating:
                        confidence += 10
                      # RSI confirmation for 5m breakdown
                    rsi_short_threshold = higher_tf_context.get("rsi_short_threshold", MAX_RSI_BEARISH)
                    if rsi_5 <= rsi_short_threshold:  # Adaptive threshold
                        confidence += 5
                    if trend_dir == "DOWN":
                        confidence += 10
                    if risk_reward >= MIN_RISK_REWARD_RATIO:
                        confidence += 5
                    # NEW: ORB boost
                    if is_orb_breakout and orb_direction == "BEARISH":
                        confidence += orb_boost

                    confidence = min(confidence, 95.0)

                    breakout_signal = Signal(
                        signal_type=SignalType.BEARISH_BREAKOUT,
                        instrument=self.instrument,
                        timeframe="5MIN",
                        price_level=breakdown_level,
                        entry_price=current_price,
                        stop_loss=sl,
                        take_profit=tp1,
                        take_profit_2=tp2,
                        take_profit_3=tp3,
                        confidence=confidence,
                        volume_confirmed=has_surge,
                        momentum_confirmed=(
                            rsi_5 <= MAX_RSI_BEARISH
                        ),
                        risk_reward_ratio=risk_reward,
                        timestamp=pd.Timestamp.now(),
                        description=(
                            f"Bearish breakdown at {breakdown_level:.2f} | "
                            f"RR: {risk_reward:.2f} | "
                            f"Consolidation: {is_consolidating}"
                            f"{' | ORB Breakout' if is_orb_breakout and orb_direction == 'BEARISH' else ''}"
                        ),
                        atr=atr,
                        debug_info={
                            "surge_ratio": surge_ratio,
                            "is_consolidating": is_consolidating,
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

    def _validate_retest_structure(
        self, 
        df: pd.DataFrame, 
        level: float, 
        direction: str
    ) -> Tuple[bool, str]:
        """
        Validate retest is actual bounce, not breakdown.
        
        Validates:
        1. Penetration depth (< 0.5 ATR below support / above resistance)
        2. Bounce strength (close must be 25%+ away from wick)
        3. Volume confirmation (> 70% of recent average)
        
        Args:
            df: OHLCV DataFrame
            level: Support/resistance level being tested
            direction: "LONG" for support retest, "SHORT" for resistance
        
        Returns:
            (is_valid, reason)
        """
        try:
            curr = df.iloc[-1]
            atr = self._calculate_atr(df)
            
            # Volume check
            if "volume" in df.columns and df["volume"].sum() > 0:
                avg_volume = df["volume"].tail(10).mean()
                if curr["volume"] < avg_volume * 0.7:
                    return False, f"Low volume: {curr['volume']:.0f} < {avg_volume*0.7:.0f}"
            
            if direction == "LONG":
                # Support retest validation
                
                # Check 1: Penetration (how far below support?)
                penetration = level - curr["low"]
                max_penetration = atr * 0.5
                
                if penetration > max_penetration:
                    return False, f"Deep penetration: {penetration:.2f} pts (max {max_penetration:.2f})"
                
                # Check 2: Bounce strength (close away from low)
                bounce_distance = curr["close"] - curr["low"]
                candle_range = curr["high"] - curr["low"]
                
                if candle_range > 0:
                    bounce_pct = (bounce_distance / candle_range) * 100
                    if bounce_pct < 25.0:
                        return False, f"Weak bounce: {bounce_pct:.0f}% (need 25%+)"
                    return True, f"Valid bounce: {bounce_pct:.0f}% from low, penetration {penetration:.2f} pts"
                return True, "Valid structure"
            
            else:  # SHORT - resistance retest
                # Same logic inverted
                penetration = curr["high"] - level
                max_penetration = atr * 0.5
                
                if penetration > max_penetration:
                    return False, f"Deep penetration: {penetration:.2f} pts"
                
                bounce_distance = curr["high"] - curr["close"]
                candle_range = curr["high"] - curr["low"]
                
                if candle_range > 0:
                    bounce_pct = (bounce_distance / candle_range) * 100
                    if bounce_pct < 25.0:
                        return False, f"Weak bounce: {bounce_pct:.0f}%"
                    return True, f"Valid bounce: {bounce_pct:.0f}% from high"
                return True, "Valid structure"
        
        except Exception as e:
            logger.warning(f"⚠️ Retest structure validation failed: {e}")
            return True, "Validation skipped (error)"  # Don't block on errors


    def detect_retest_setup(
        self, 
        df: pd.DataFrame, 
        support_resistance: TechnicalLevels,
        higher_tf_context: Dict = None
    ) -> Optional[Signal]:
        """
        Detect retest setup with role reversal logic.
        
        Properly identifies:
        - Support retest: Price above level (or broke above and pulled back)
        - Resistance retest: Price below level (or broke below and bounced)
        - Role reversal: Former resistance becomes support after breakout
        """
        try:
            if df is None or df.empty or len(df) < 10:
                return None

            current = df.iloc[-1]
            current_price = float(current["close"])
            
            # Look back to check for recent breakouts (Increased to 50 bars / ~4 hours)
            # to capture breakouts that happened earlier in the session
            recent_bars = df.tail(50)
            recent_highs = recent_bars["high"].values
            recent_lows = recent_bars["low"].values

            candidate_levels = (
                support_resistance.support_levels[:3]
                + support_resistance.resistance_levels[:3]
            )

            for level in candidate_levels:
                distance_pct = abs(current_price - level) / level * 100.0
                if distance_pct <= RETEST_ZONE_PERCENT:
                    
                    # ====================
                    # Determine Role Reversal
                    # ====================
                    # Check if price recently broke through this level
                    broke_above = any(recent_highs > level * 1.001)  # 0.1% buffer
                    broke_below = any(recent_lows < level * 0.999)   # 0.1% buffer
                    
                    # Current position
                    price_above_level = current_price > level
                    
                    # ====================
                    # Retest Logic
                    # ====================
                    signal_type = None
                    description = ""
                    is_long = False
                    
                    # SUPPORT RETEST scenarios:
                    # 1. Price above level AND broke above recently (pullback to new support)
                    # 2. Price hovering at support from below
                    if price_above_level:
                        if broke_above:
                            # Role reversal: former resistance now support
                            signal_type = SignalType.SUPPORT_BOUNCE
                            description = f"Support retest at {level:.2f} (former resistance, role reversal)"
                            is_long = True
                        else:
                            # Regular support bounce
                            signal_type = SignalType.SUPPORT_BOUNCE
                            description = f"Support retest at {level:.2f}"
                            is_long = True
                    
                    # RESISTANCE RETEST scenarios:
                    # 1. Price below level AND broke below recently (bounce to new resistance)
                    # 2. Price testing resistance from below
                    else:  # price_above_level == False
                        if broke_below:
                            # Role reversal: former support now resistance
                            signal_type = SignalType.RESISTANCE_BOUNCE
                            description = f"Resistance retest at {level:.2f} (former support, role reversal)"
                            is_long = False
                        else:
                            # Regular resistance test
                            signal_type = SignalType.RESISTANCE_BOUNCE
                            description = f"Resistance retest at {level:.2f}"
                            is_long = False
                    
                    if signal_type is None:
                        continue
                    
                    logger.info(
                        f"🎯 RETEST SETUP | {description} | "
                        f"Price: {current_price:.2f} | Level: {level:.2f} | "
                        f"Dist: {abs(current_price - level):.2f} pts ({distance_pct:.3f}%)"
                    )

                    # NEW: Validate bounce structure
                    direction_for_validation = "LONG" if is_long else "SHORT"
                    is_valid, validation_reason = self._validate_retest_structure(
                        df, level, direction_for_validation
                    )
                    
                    if not is_valid:
                        logger.info(
                            f"⏭️ Retest rejected: {validation_reason} | "
                            f"Level: {level:.2f}, Direction: {direction_for_validation}"
                        )
                        continue  # Skip this signal, try next level
                    
                    logger.info(f"✅ Retest validated: {validation_reason}")

                    atr = support_resistance.atr
                    
                    # ====================
                    # Entry, SL, TP Logic
                    # ====================
                    
                    # Determine Strong Trend
                    trend_15m = higher_tf_context.get("trend_direction", "FLAT") if higher_tf_context else "FLAT"
                    rsi_15 = float(higher_tf_context.get("rsi_15", 50)) if higher_tf_context else 50.0
                    
                    is_strong_trend = False
                    if is_long:
                        is_strong_trend = trend_15m == "UP" and rsi_15 >= 55
                    else:
                        is_strong_trend = trend_15m == "DOWN" and rsi_15 <= 45

                    # ====================
                    # Entry, SL, TP Logic
                    # ====================
                    if is_long:
                        # LONG: Support retest
                        entry_price = current_price
                        stop_loss = level - (atr * 0.5)  # SL below support
                        
                        tp1, tp2, tp3 = self._calculate_multi_targets(
                            entry_price, stop_loss, "LONG", support_resistance, is_strong_trend
                        )
                    
                    else:
                        # SHORT: Resistance retest
                        entry_price = current_price
                        stop_loss = level + (atr * 0.5)  # SL above resistance
                        
                        tp1, tp2, tp3 = self._calculate_multi_targets(
                            entry_price, stop_loss, "SHORT", support_resistance, is_strong_trend
                        )
                    
                    # Calculate R:R using T1
                    risk = abs(entry_price - stop_loss)
                    reward = abs(tp1 - entry_price)
                    rr = reward / risk if risk > 0 else 0
                    
                    # Skip if R:R too low
                    if round(rr, 2) < MIN_RISK_REWARD_RATIO:
                        logger.info(f"⏭️ Skipping retest - poor R:R ({rr:.2f}) < {MIN_RISK_REWARD_RATIO}")
                        continue

                    # ====================
                    # Dynamic Confidence Scoring
                    # ====================
                    confidence = 50.0  # Base Score
                    
                    if higher_tf_context:
                        trend_15m = higher_tf_context.get("trend_direction", "FLAT")
                        rsi_15 = float(higher_tf_context.get("rsi_15", 50))
                        price_above_vwap = higher_tf_context.get("price_above_vwap", False)
                        price_above_ema20 = higher_tf_context.get("price_above_ema20", False)
                        
                        # 1. Trend Alignment (+10%)
                        if (is_long and trend_15m == "UP") or \
                           (not is_long and trend_15m == "DOWN"):
                            confidence += 10
                            
                        # 2. Moving Average Support (+5%)
                        if (is_long and price_above_vwap) or \
                           (not is_long and not price_above_vwap):
                            confidence += 5
                            
                        # 3. RSI Confirmation (+5%)
                        if (is_long and rsi_15 > 50) or (not is_long and rsi_15 < 50):
                            confidence += 5
                            
                        # 4. Key Level Confluence (+5%)
                        # Check if level matches PDH/PDL or ORB High/Low
                        pdh = support_resistance.pdh
                        pdl = support_resistance.pdl
                        orb_high = higher_tf_context.get("orb_high", 0)
                        orb_low = higher_tf_context.get("orb_low", 0)
                        
                        confluence_level = False
                        for key_lvl in [pdh, pdl, orb_high, orb_low]:
                            if key_lvl > 0 and abs(level - key_lvl) < (level * 0.002):
                                confluence_level = True
                                break
                        
                        if confluence_level:
                            confidence += 5

                    # 5. Role Reversal (+10%) - Stronger than simple touch
                    if (is_long and description and "role reversal" in description) or \
                       (not is_long and description and "role reversal" in description):
                        confidence += 10
                        
                    # 6. High R:R (+5%)
                    if rr >= 2.0:
                        confidence += 5
                        
                    confidence = min(confidence, 95.0)

                    return Signal(
                        signal_type=signal_type,
                        instrument=self.instrument,
                        timeframe="5MIN",
                        price_level=level,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=tp1,
                        take_profit_2=tp2,
                        take_profit_3=tp3,
                        confidence=confidence,
                        volume_confirmed=False,
                        momentum_confirmed=False,
                        risk_reward_ratio=rr,
                        timestamp=pd.Timestamp.now(),
                        description=description,
                        atr=atr,
                        debug_info={
                            "distance_pct": distance_pct,
                            "broke_above": broke_above,
                            "broke_below": broke_below,
                            "price_above_level": price_above_level,
                            "is_role_reversal": "role reversal" in description,
                            "is_strong_trend": is_strong_trend
                        },
                    )

            return None

        except Exception as e:
            logger.error(f"❌ Retest detection failed: {str(e)}")
            return None

    def detect_inside_bar(
        self,
        df: pd.DataFrame,
        higher_tf_context: Dict,
        support_resistance: TechnicalLevels
    ) -> Optional[Signal]:
        """
        Detect high-probability inside bar setup with full context awareness.
        
        Only fires when:
        - Pattern is valid (size, volume)
        - Trend alignment (15m + VWAP + 20 EMA + prev day)
        - Logical target with good R:R
        - Breakout confirmed (price beyond mother bar range)
        
        Args:
            df: 5m OHLCV data
            higher_tf_context: Contains 15m trend, VWAP, 20 EMA, prev day trend
            support_resistance: S/R levels for smart targeting
        """
        try:
            if df is None or len(df) < 2:
                return None

            prev_candle = df.iloc[-2]  # Mother bar
            curr_candle = df.iloc[-1]  # Inside bar (current)

            # ====================
            # 1. Pattern Detection
            # ====================
            is_inside = (
                curr_candle["high"] < prev_candle["high"]
                and curr_candle["low"] > prev_candle["low"]
            )

            if not is_inside:
                return None

            logger.info("📊 INSIDE BAR PATTERN DETECTED | Validating setup...")

            # ====================
            # 2. Pattern Quality Check
            # ====================
            is_valid, rejection_reason = self._validate_inside_bar_pattern(
                prev_candle, curr_candle, df
            )
            
            if not is_valid:
                logger.info(f"⏭️ Inside bar rejected: {rejection_reason}")
                return None

            # ====================
            # 3. Extract Context
            # ====================
            trend_15m = higher_tf_context.get("trend_direction", "FLAT")
            price_above_vwap = higher_tf_context.get("price_above_vwap", False)
            price_above_ema20 = higher_tf_context.get("price_above_ema20", False)
            vwap_slope = higher_tf_context.get("vwap_slope", "FLAT")
            prev_day_trend = higher_tf_context.get("prev_day_trend", "FLAT")
            vwap_5m = higher_tf_context.get("vwap_5m", 0.0)
            ema_20_5m = higher_tf_context.get("ema_20_5m", 0.0)
            
            current_price = float(curr_candle["close"])
            rsi_5 = self._calculate_rsi(df)

            # ====================
            # 4. Determine Directional Bias
            # ====================
            direction = None
            signal_type = None
            
            # LONG bias conditions
            long_conditions = [
                price_above_vwap,
                price_above_ema20,
                trend_15m == "UP" or prev_day_trend == "UP",
            ]
            
            # SHORT bias conditions
            short_conditions = [
                not price_above_vwap,
                not price_above_ema20,
                trend_15m == "DOWN" or prev_day_trend == "DOWN",
            ]
            
            # Require at least 2 of 3 conditions
            if sum(long_conditions) >= 2:
                direction = "LONG"
                signal_type = SignalType.INSIDE_BAR
            elif sum(short_conditions) >= 2:
                direction = "SHORT"
                signal_type = SignalType.INSIDE_BAR
            else:
                logger.info(
                    "⏭️ Inside bar skipped - no clear directional bias | "
                    f"VWAP: {price_above_vwap} | 20EMA: {price_above_ema20} | "
                    f"Trend15m: {trend_15m} | PrevDay: {prev_day_trend}"
                )
                return None

            logger.info(f"✅ Directional bias: {direction}")

            # ====================
            # 5. Breakout Entry Logic
            # ====================
            # Check if breakout has actually happened
            if direction == "LONG":
                # For LONG, need close ABOVE mother bar high
                if current_price <= prev_candle["high"]:
                    logger.debug(
                        "⏳ Inside bar LONG setup pending - waiting for breakout above mother bar high"
                    )
                    return None  # Wait for actual breakout
                
                entry_price = float(prev_candle["high"])
                sl_price = float(prev_candle["low"])
                
            else:  # SHORT
                # For SHORT, need close BELOW mother bar low
                if current_price >= prev_candle["low"]:
                    logger.debug(
                        "⏳ Inside bar SHORT setup pending - waiting for breakout below mother bar low"
                    )
                    return None  # Wait for actual breakout
                
                entry_price = float(prev_candle["low"])
                sl_price = float(prev_candle["high"])

            # Check if mother bar is too wide - use 50% level for SL
            mother_range = abs(prev_candle["high"] - prev_candle["low"])
            atr = support_resistance.atr
            if mother_range > atr * 2.0:
                logger.info(f"⚠️ Wide mother bar detected - using 50% SL level")
                if direction == "LONG":
                    sl_price = prev_candle["low"] + (mother_range * 0.5)
                else:
                    sl_price = prev_candle["high"] - (mother_range * 0.5)

            # ====================
            # 6. Volume Confirmation
            # ====================
            vol_confirmed, vol_ratio, _ = self.check_volume_confirmation(df)

            # ====================
            # 7. Smart Target Calculation
            # ====================
            is_strong_trend = vol_confirmed
            
            tp1, tp2, tp3 = self._calculate_multi_targets(
                entry_price, sl_price, direction, support_resistance, is_strong_trend
            )
            
            risk = abs(entry_price - sl_price)
            reward = abs(tp1 - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            if tp1 == 0 or rr_ratio < 1.0:
                logger.info(f"⏭️ Inside bar skipped - no valid target or poor R:R ({rr:.2f} < {MIN_RISK_REWARD_RATIO})")
                return None

            # ====================
            # 8. Confidence Scoring
            # ====================
            confidence = 60.0
            
            # VWAP alignment
            if (direction == "LONG" and price_above_vwap) or \
               (direction == "SHORT" and not price_above_vwap):
                confidence += 10
                
            # 20 EMA alignment
            if (direction == "LONG" and price_above_ema20) or \
               (direction == "SHORT" and not price_above_ema20):
                confidence += 10
                
            # 15m trend alignment
            if (direction == "LONG" and trend_15m == "UP") or \
               (direction == "SHORT" and trend_15m == "DOWN"):
                confidence += 10
                
            # Previous day trend alignment
            if (direction == "LONG" and prev_day_trend == "UP") or \
               (direction == "SHORT" and prev_day_trend == "DOWN"):
                confidence += 5
                
            # Volume confirmation
            if vol_confirmed:
                confidence += 10
                
            # Good R:R
            if rr_ratio >= 2.0:
                confidence += 10
                
            # RSI alignment
            if (direction == "LONG" and rsi_5 >= 50) or \
               (direction == "SHORT" and rsi_5 <= 50):
                confidence += 5

            confidence = min(confidence, 95.0)

            # ====================
            # 9. Build Description
            # ====================
            description = (
                f"Inside bar {direction} breakout | "
                f"VWAP: {'✓' if ((direction=='LONG' and price_above_vwap) or (direction=='SHORT' and not price_above_vwap)) else '✗'} | "
                f"20EMA: {'✓' if ((direction=='LONG' and price_above_ema20) or (direction=='SHORT' and not price_above_ema20)) else '✗'} | "
                f"15m: {trend_15m} | PrevDay: {prev_day_trend}"
            )

            logger.info(
                f"🎯 HIGH-QUALITY INSIDE BAR {direction} | "
                f"Entry: {entry_price:.2f} | SL: {sl_price:.2f} | T1: {tp1:.2f} | "
                f"R:R: {rr_ratio:.2f} | Conf: {confidence:.0f}%"
            )

            return Signal(
                signal_type=signal_type,
                instrument=self.instrument,
                timeframe="5MIN",
                price_level=float(curr_candle["close"]),
                entry_price=entry_price,
                stop_loss=sl_price,
                take_profit=tp1,
                take_profit_2=tp2,
                take_profit_3=tp3,
                confidence=confidence,
                volume_confirmed=vol_confirmed,
                momentum_confirmed=(rsi_5 >= 50 if direction == "LONG" else rsi_5 <= 50),
                risk_reward_ratio=rr_ratio,
                timestamp=pd.Timestamp.now(),
                description=description,
                debug_info={
                    "direction": direction,
                    "mother_high": float(prev_candle["high"]),
                    "mother_low": float(prev_candle["low"]),
                    "inside_high": float(curr_candle["high"]),
                    "inside_low": float(curr_candle["low"]),
                    "trend_15m": trend_15m,
                    "vwap_5m": vwap_5m,
                    "ema_20_5m": ema_20_5m,
                    "prev_day_trend": prev_day_trend,
                    "rsi_5": rsi_5,
                    "vol_ratio": vol_ratio,
                    "is_strong_trend": is_strong_trend
                },
            )

        except Exception as e:
            logger.error(f"❌ Inside bar detection failed: {str(e)}")
            return None

    def _validate_inside_bar_pattern(
        self,
        prev_candle: pd.Series,
        curr_candle: pd.Series,
        df: pd.DataFrame
    ) -> Tuple[bool, str]:
        """
        Validate inside bar pattern quality.
        
        Returns:
            (is_valid, rejection_reason)
        """
        try:
            mother_range = prev_candle['high'] - prev_candle['low']
            inside_range = curr_candle['high'] - curr_candle['low']
            
            # Check if inside bar is not too tiny
            if inside_range < mother_range * 0.2:
                return False, "Inside bar too small (< 20% of mother bar)"
            
            # Check if inside bar is not too large  
            if inside_range > mother_range * 0.8:
                return False, "Inside bar too large (> 80% of mother bar)"
            
            # Check if mother bar is not too wide
            atr = self._calculate_atr(df)
            if mother_range > atr * 2.5:
                return False, f"Mother bar too wide ({mother_range:.2f} > 2.5x ATR)"
            
            # Check volume (if available)
            if df['volume'].sum() > 0:
                avg_vol = df['volume'].tail(20).mean()
                mother_vol = prev_candle['volume']
                curr_vol = curr_candle['volume']
                
                # Prefer volume on mother bar or breakout bar
                if mother_vol < avg_vol * 0.7 and curr_vol < avg_vol * 0.7:
                    return False, "Low volume on both mother and inside bar"
            
            return True, ""
            
        except Exception as e:
            logger.warning(f"⚠️ Pattern validation failed: {str(e)}")
            return True, ""  # Default to valid if check fails

    def _calculate_inside_bar_targets(
        self,
        entry: float,
        sl: float,
        direction: str,
        support_resistance: TechnicalLevels,
        current_price: float
    ) -> Tuple[Optional[float], float, str]:
        """
        Calculate intelligent take profit using S/R and PDH/PDL.
        
        Args:
            entry: Entry price
            sl: Stop loss
            direction: "LONG" or "SHORT"
            support_resistance: S/R levels
            current_price: Current market price
            
        Returns:
            (take_profit, risk_reward_ratio, target_reason)
        """
        try:
            risk = abs(entry - sl)
            atr = support_resistance.atr
            
            # Find nearest logical target
            target = None
            target_reason = ""
            
            if direction == "LONG":
                # Look for resistance above current price
                candidates = []
                
                # Check resistance clusters
                for r in support_resistance.resistance_levels:
                    if r > current_price:
                        candidates.append((r, "Resistance"))
                
                # Check PDH
                if support_resistance.pdh > current_price:
                    candidates.append((support_resistance.pdh, "PDH"))
                
                # Sort by distance and pick nearest
                if candidates:
                    candidates.sort(key=lambda x: abs(x[0] - current_price))
                    target, target_reason = candidates[0]
                else:
                    # Fallback to ATR-based
                    target = entry + (atr * 2.0)
                    target_reason = "ATR 2.0x"
                    
            else:  # SHORT
                # Look for support below current price
                candidates = []
                
                # Check support clusters
                for s in support_resistance.support_levels:
                    if s < current_price:
                        candidates.append((s, "Support"))
                
                # Check PDL
                if support_resistance.pdl < current_price and support_resistance.pdl > 0:
                    candidates.append((support_resistance.pdl, "PDL"))
                
                # Sort by distance and pick nearest
                if candidates:
                    candidates.sort(key=lambda x: abs(x[0] - current_price))
                    target, target_reason = candidates[0]
                else:
                    # Fallback to ATR-based
                    target = entry - (atr * 2.0)
                    target_reason = "ATR 2.0x"
            
            # Calculate R:R
            if target:
                reward = abs(target - entry)
                rr = reward / risk if risk > 0 else 0
                
                # Skip if R:R too low
                if rr < 1.5:
                    logger.info(f"⏭️ Skipping inside bar - poor R:R ({rr:.2f} < {MIN_RISK_REWARD_RATIO})")
                    return None, 0.0, "R:R < 1.5"
                
                return target, rr, target_reason
            
            return None, 0.0, "No logical target found"
            
        except Exception as e:
            logger.warning(f"⚠️ Target calculation failed: {str(e)}")
            # Fallback to simple ATR-based
            if direction == "LONG":
                target = entry + (risk * 2.0)
            else:
                target = entry - (risk * 2.0)
            return target, 2.0, "ATR Fallback"

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

    def _calculate_rsi_series(self, df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
        """Calculate full RSI series."""
        try:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0.0).rolling(period).mean()
            rs = gain / loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi.fillna(50.0)
        except Exception:
            return pd.Series(50.0, index=df.index)

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

    def _calculate_macd(
        self, 
        df: pd.DataFrame, 
        fast: int = None, 
        slow: int = None, 
        signal: int = None
    ) -> Dict:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            df: OHLCV DataFrame
            fast: Fast EMA period (default from config)
            slow: Slow EMA period (default from config)
            signal: Signal line period (default from config)
        
        Returns:
            {
                "macd_line": float,
                "signal_line": float,
                "histogram": float,
                "crossover": str  # "BULLISH", "BEARISH", "NONE"
            }
        """
        try:
            from config.settings import MACD_FAST, MACD_SLOW, MACD_SIGNAL
            
            fast = fast or MACD_FAST
            slow = slow or MACD_SLOW
            signal = signal or MACD_SIGNAL
            
            if len(df) < slow + signal:
                logger.warning(f"Insufficient data for MACD calculation (need {slow + signal} candles)")
                return {
                    "macd_line": 0.0,
                    "signal_line": 0.0,
                    "histogram": 0.0,
                    "crossover": "NONE"
                }
            
            # Calculate MACD line
            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            # Detect crossover (check last 2 candles)
            crossover = "NONE"
            if len(histogram) >= 2:
                prev_hist = histogram.iloc[-2]
                curr_hist = histogram.iloc[-1]
                
                if prev_hist <= 0 and curr_hist > 0:
                    crossover = "BULLISH"
                    logger.info(f"🔼 MACD Bullish Crossover: Histogram {prev_hist:.2f} → {curr_hist:.2f}")
                elif prev_hist >= 0 and curr_hist < 0:
                    crossover = "BEARISH"
                    logger.info(f"🔽 MACD Bearish Crossover: Histogram {prev_hist:.2f} → {curr_hist:.2f}")
            
            return {
                "macd_line": round(macd_line.iloc[-1], 2),
                "signal_line": round(signal_line.iloc[-1], 2),
                "histogram": round(histogram.iloc[-1], 2),
                "crossover": crossover
            }
        
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {
                "macd_line": 0.0,
                "signal_line": 0.0,
                "histogram": 0.0,
                "crossover": "NONE"
            }

    def detect_ema_crossover(
        self, 
        df: pd.DataFrame, 
        fast: int = None, 
        slow: int = None
    ) -> Dict:
        """
        Detect EMA crossover for directional bias.
        
        Args:
            df: OHLCV DataFrame
            fast: Fast EMA period (default from config)
            slow: Slow EMA period (default from config)
        
        Returns:
            {
                "bias": str,  # "BULLISH", "BEARISH", "NEUTRAL"
                "confidence": float,  # 0.0 to 1.0
                "price_separation_pct": float,
                "ema_fast": float,
                "ema_slow": float
            }
        """
        try:
            from config.settings import EMA_CROSSOVER_FAST, EMA_CROSSOVER_SLOW
            
            fast = fast or EMA_CROSSOVER_FAST
            slow = slow or EMA_CROSSOVER_SLOW
            
            if len(df) < slow + 5:
                logger.warning(f"Insufficient data for EMA crossover (need {slow + 5} candles)")
                return {
                    "bias": "NEUTRAL",
                    "confidence": 0.0,
                    "price_separation_pct": 0.0,
                    "ema_fast": 0.0,
                    "ema_slow": 0.0
                }
            
            # Calculate EMAs
            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
            
            current_price = df['close'].iloc[-1]
            ema_f = ema_fast.iloc[-1]
            ema_s = ema_slow.iloc[-1]
            
            # Check for crossover
            bias = "NEUTRAL"
            if len(ema_fast) >= 2:
                prev_f = ema_fast.iloc[-2]
                prev_s = ema_slow.iloc[-2]
                
                # Bullish crossover: Fast EMA crosses above Slow EMA + Price > Fast EMA
                if prev_f <= prev_s and ema_f > ema_s and current_price > ema_f:
                    bias = "BULLISH"
                    logger.info(f"🔼 EMA {fast}/{slow} Bullish Crossover detected")
                
                # Bearish crossover: Fast EMA crosses below Slow EMA + Price < Fast EMA
                elif prev_f >= prev_s and ema_f < ema_s and current_price < ema_f:
                    bias = "BEARISH"
                    logger.info(f"🔽 EMA {fast}/{slow} Bearish Crossover detected")
            
            # Calculate price separation percentage (for confidence)
            price_sep_pct = abs(current_price - ema_s) / ema_s * 100
            
            # Confidence: Higher if price is well separated from slow EMA
            # 1% separation = 100% confidence
            confidence = min(price_sep_pct / 1.0, 1.0)
            
            return {
                "bias": bias,
                "confidence": round(confidence, 2),
                "price_separation_pct": round(price_sep_pct, 2),
                "ema_fast": round(ema_f, 2),
                "ema_slow": round(ema_s, 2)
            }
        
        except Exception as e:
            logger.error(f"Error detecting EMA crossover: {e}")
            return {
                "bias": "NEUTRAL",
                "confidence": 0.0,
                "price_separation_pct": 0.0,
                "ema_fast": 0.0,
                "ema_slow": 0.0
            }

    def _calculate_vwap(self, df: pd.DataFrame) -> Tuple[pd.Series, float, str]:
        """
        Calculate intraday VWAP (Volume Weighted Average Price).
        Resets at day boundaries for intraday analysis.
        
        Returns:
            (vwap_series, current_vwap, vwap_slope)
        """
        try:
            df = df.copy()
            
            # Check if we have volume data
            if df['volume'].sum() == 0:
                logger.debug("⚠️ No volume data for VWAP - using simple average")
                vwap_series = df['close']
                return vwap_series, float(vwap_series.iloc[-1]), "FLAT"
            
            # Calculate typical price
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
            
            # Identify day boundaries (reset VWAP at each day)
            df['date'] = pd.to_datetime(df.index).date
            
            # Calculate VWAP for each day
            vwap_list = []
            for date, group in df.groupby('date'):
                group = group.copy()
                group['cum_vol'] = group['volume'].cumsum()
                group['cum_vol_price'] = (group['typical_price'] * group['volume']).cumsum()
                group['vwap'] = group['cum_vol_price'] / group['cum_vol']
                vwap_list.append(group['vwap'])
            
            vwap_series = pd.concat(vwap_list)
            current_vwap = float(vwap_series.iloc[-1])
            
            # Determine VWAP slope (last 5 bars)
            if len(vwap_series) >= 5:
                vwap_slope = "UP" if vwap_series.iloc[-1] > vwap_series.iloc[-5] else "DOWN"
            else:
                vwap_slope = "FLAT"
            
            logger.debug(f"VWAP: {current_vwap:.2f} | Slope: {vwap_slope}")
            return vwap_series, current_vwap, vwap_slope
            
        except Exception as e:
            logger.warning(f"⚠️ VWAP calculation failed: {str(e)}")
            # Fallback to close price
            return df['close'], float(df['close'].iloc[-1]), "FLAT"

    def _get_previous_day_trend(self, df: pd.DataFrame) -> str:
        """
        Determine previous day's trend based on daily data.
        
        Args:
            df: DataFrame with at least 2 days of data
            
        Returns:
            "UP" / "DOWN" / "FLAT"
        """
        try:
            if df is None or len(df) < 2:
                return "FLAT"
            
            # Get yesterday's candle (second to last)
            prev_day = df.iloc[-2]
            
            # Simple trend: close vs open
            if prev_day['close'] > prev_day['open'] * 1.002:  # 0.2% threshold
                return "UP"
            elif prev_day['close'] < prev_day['open'] * 0.998:
                return "DOWN"
            else:
                return "FLAT"
                
        except Exception as e:
            logger.warning(f"⚠️ Previous day trend calc failed: {str(e)}")
            return "FLAT"

    def detect_pin_bar(
        self,
        df: pd.DataFrame,
        support_resistance: TechnicalLevels,
        higher_tf_context: Dict,
    ) -> Optional[Signal]:
        """
        Detect pin bar (rejection candle with long wick).
        
        Bullish Pin Bar (Hammer):
            - Long lower wick (>60% of range)
            - Small body (<30% of range)
            - At support level
            - Uptrend or reversal setup
        
        Bearish Pin Bar (Shooting Star):
            - Long upper wick (>60% of range)
            - Small body (<30% of range)
            - At resistance level
            - Downtrend or reversal setup
        """
        try:
            if df is None or df.empty or len(df) < 20:
                return None

            current = df.iloc[-1]
            current_open = float(current["open"])
            current_close = float(current["close"])
            current_high = float(current["high"])
            current_low = float(current["low"])
            current_price = current_close

            # Calculate candle components
            body_size = abs(current_close - current_open)
            upper_wick = current_high - max(current_open, current_close)
            lower_wick = min(current_open, current_close) - current_low
            total_range = current_high - current_low

            if total_range < 0.0001:  # Avoid division by zero
                return None

            body_pct = (body_size / total_range)
            upper_wick_pct = (upper_wick / total_range)
            lower_wick_pct = (lower_wick / total_range)

            atr = support_resistance.atr
            trend_dir = higher_tf_context.get("trend_direction", "FLAT")
            rsi_15 = float(higher_tf_context.get("rsi_15", 50.0))

            # =============================
            # BULLISH PIN BAR (Hammer)
            # =============================
            if lower_wick_pct > 0.6 and body_pct < 0.3:
                # Check if at support level
                at_support = False
                support_level = None
                
                for level in support_resistance.support_levels[:3]:
                    distance_pct = abs(current_low - level) / level * 100
                    if distance_pct <= RETEST_ZONE_PERCENT:  # Use global setting (0.15% = ~35-40 pts for Nifty)
                        at_support = True
                        support_level = level
                        break
                
                # Also check if near PDL
                if not at_support and support_resistance.pdl > 0:
                    distance_pct = abs(current_low - support_resistance.pdl) / support_resistance.pdl * 100
                    if distance_pct <= RETEST_ZONE_PERCENT:
                        at_support = True
                        support_level = support_resistance.pdl

                if at_support:
                    # Logic Update: Allow counter-trend IF confirmed by RSI/Volume
                    is_valid_setup = False
                    setup_reason = ""
                    
                    if DEBUG_MODE:
                         logger.debug(f"DEBUG: PinBar at Support. Trend: {trend_dir}, RSI: {rsi_15}")
                    
                    if trend_dir in ["UP", "FLAT"]:
                        is_valid_setup = True
                        setup_reason = "Trend Alignment"
                    elif trend_dir == "DOWN":
                        # Counter-trend checks - use adaptive threshold
                        # Require Oversold RSI OR Volume Surge
                        rsi_short_threshold = higher_tf_context.get("rsi_short_threshold", MAX_RSI_BEARISH)
                        vol_surge, _, _ = self._detect_volume_surge(df)
                        if rsi_15 < rsi_short_threshold:  # Adaptive: normally 40, but 35 in high volatility
                            is_valid_setup = True
                            setup_reason = f"Counter-Trend (RSI {rsi_15:.1f} Oversold)"
                        elif vol_surge:
                            is_valid_setup = True
                            setup_reason = "Counter-Trend (Volume Surge)"
                            
                    if not is_valid_setup:
                         if DEBUG_MODE:
                             logger.debug(f"DEBUG: Setup Invalid. RSI {rsi_15} not < 40 and No Surge")

                    if is_valid_setup:
                        logger.info(
                            f"🔨 BULLISH PIN BAR detected | {setup_reason} | "
                            f"Lower wick: {lower_wick_pct*100:.1f}% | "
                            f"At support: {support_level:.2f}"
                        )

                        entry_price = current_close
                        stop_loss = current_low - (atr * 0.5)
                        
                        # Calculate risk first
                        risk = abs(entry_price - stop_loss)
                        
                        # Target: Next resistance or PDH, but cap at 3x risk (realistic for intraday)
                        target = None
                        for r in support_resistance.resistance_levels:
                            if r > current_price:
                                target = r
                                break
                        
                        if not target and support_resistance.pdh > current_price:
                            target = support_resistance.pdh
                        
                        if not target:
                            target = current_price + (risk * 3.0)  # Default: 3x risk
                        else:
                            # Cap target at 3x risk to prevent unrealistic R:R
                            max_target = entry_price + (risk * 3.0)
                            if target > max_target:
                                logger.info(f"📉 Target capped from {target:.2f} to {max_target:.2f} (3x risk)")
                                target = max_target
                        
                        reward = abs(target - entry_price)
                        rr = reward / risk if risk > 0 else 0
                        
                        if rr < 1.5:
                            logger.info(f"⏭️ Bullish pin bar skipped: Poor R:R ({rr:.2f})")
                            return None

                        confidence = 65.0
                        if trend_dir == "UP":
                            confidence += 10
                        if rsi_15 < 40:  # Deep Oversold
                            confidence += 10
                        elif rsi_15 < 50:
                            confidence += 5
                        if lower_wick_pct > 0.7:  # Very long wick
                            confidence += 5

                        return Signal(
                            signal_type=SignalType.BULLISH_PIN_BAR,
                            instrument=self.instrument,
                            timeframe="5MIN",
                            price_level=support_level,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=target,
                            confidence=min(confidence, 90.0),
                            volume_confirmed=False,
                            momentum_confirmed=(rsi_15 < 50),
                            risk_reward_ratio=rr,
                            timestamp=pd.Timestamp.now(),
                            description=(
                                f"Bullish pin bar (Entry: {entry_price:.2f}, Near Support: {support_level:.2f}) | {setup_reason} | "
                                f"Lower wick: {lower_wick_pct*100:.0f}% | RR: {rr:.1f}"
                            ),
                            debug_info={
                                "lower_wick_pct": lower_wick_pct,
                                "body_pct": body_pct,
                                "trend_dir": trend_dir,
                                "rsi_15": rsi_15,
                            },
                        )

            # =============================
            # BEARISH PIN BAR (Shooting Star)
            # =============================
            if upper_wick_pct > 0.6 and body_pct < 0.3:
                # Check if at resistance level
                at_resistance = False
                resistance_level = None
                
                for level in support_resistance.resistance_levels[:3]:
                    distance_pct = abs(current_high - level) / level * 100
                    if distance_pct <= RETEST_ZONE_PERCENT:  # Within 0.5% of resistance
                        at_resistance = True
                        resistance_level = level
                        break
                
                # Also check if near PDH
                if not at_resistance and support_resistance.pdh > 0:
                    distance_pct = abs(current_high - support_resistance.pdh) / support_resistance.pdh * 100
                    if distance_pct <= RETEST_ZONE_PERCENT:
                        at_resistance = True
                        resistance_level = support_resistance.pdh

                if at_resistance:
                    # Logic Update: Allow counter-trend IF confirmed by RSI/Volume
                    is_valid_setup = False
                    setup_reason = ""
                    
                    if trend_dir in ["DOWN", "FLAT"]:
                        is_valid_setup = True
                        setup_reason = "Trend Alignment"
                    elif trend_dir == "UP":
                        # Counter-trend bearish pin bar - use adaptive threshold
                        rsi_long_threshold = higher_tf_context.get("rsi_long_threshold", MIN_RSI_BULLISH)
                        vol_surge, _, _ = self._detect_volume_surge(df)
                        if rsi_15 > rsi_long_threshold:  # Adaptive: normally 60, but 65 in high volatility
                            is_valid_setup = True
                            setup_reason = f"Counter-Trend (RSI {rsi_15:.1f} Overbought)"
                        elif vol_surge:
                            is_valid_setup = True
                            setup_reason = "Counter-Trend (Volume Surge)"

                    if is_valid_setup:
                        logger.info(
                            f"⭐ BEARISH PIN BAR detected | {setup_reason} | "
                            f"Upper wick: {upper_wick_pct*100:.1f}% | "
                            f"At resistance: {resistance_level:.2f}"
                        )

                        entry_price = current_close
                        stop_loss = current_high + (atr * 0.5)
                        
                        # Calculate risk first
                        risk = abs(stop_loss - entry_price)
                        
                        # Target: Next support or PDL, but cap at 3x risk (realistic for intraday)
                        target = None
                        for s in support_resistance.support_levels:
                            if s < current_price:
                                target = s
                                break
                        
                        if not target and support_resistance.pdl > 0 and support_resistance.pdl < current_price:
                            target = support_resistance.pdl
                        
                        if not target:
                            target = current_price - (risk * 3.0)  # Default: 3x risk
                        else:
                            # Cap target at 3x risk to prevent unrealistic R:R
                            min_target = entry_price - (risk * 3.0)
                            if target < min_target:
                                logger.info(f"📈 Target capped from {target:.2f} to {min_target:.2f} (3x risk)")
                                target = min_target
                        
                        reward = abs(entry_price - target)
                        rr = reward / risk if risk > 0 else 0
                        
                        if rr < 1.5:
                            logger.info(f"⏭️ Bearish pin bar skipped: Poor R:R ({rr:.2f})")
                            return None

                        confidence = 65.0
                        if trend_dir == "DOWN":
                            confidence += 10
                        if rsi_15 > 60:  # Deep Overbought
                            confidence += 10
                        elif rsi_15 > 50:
                            confidence += 5
                        if upper_wick_pct > 0.7:  # Very long wick
                            confidence += 5

                        return Signal(
                            signal_type=SignalType.BEARISH_PIN_BAR,
                            instrument=self.instrument,
                            timeframe="5MIN",
                            price_level=resistance_level,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=target,
                            confidence=min(confidence, 90.0),
                            volume_confirmed=False,
                            momentum_confirmed=(rsi_15 > 50),
                            risk_reward_ratio=rr,
                            timestamp=pd.Timestamp.now(),
                            description=(
                                f"Bearish pin bar (Entry: {entry_price:.2f}, Near Resistance: {resistance_level:.2f}) | {setup_reason} | "
                                f"Upper wick: {upper_wick_pct*100:.0f}% | RR: {rr:.1f}"
                            ),
                            debug_info={
                                "upper_wick_pct": upper_wick_pct,
                                "body_pct": body_pct,
                                "trend_dir": trend_dir,
                                "rsi_15": rsi_15,
                            },
                        )

            return None

        except Exception as e:
            logger.error(f"❌ Pin bar detection failed: {str(e)}")
            return None

    def detect_engulfing(
        self,
        df: pd.DataFrame,
        support_resistance: TechnicalLevels,
        higher_tf_context: Dict,
    ) -> Optional[Signal]:
        """
        Detect bullish or bearish engulfing candlestick pattern.
        
        Bullish Engulfing:
            - Previous candle is bearish (red)
            - Current candle is bullish (green)
            - Current candle body completely engulfs previous body
            - At support level or in uptrend
            - Volume confirmation
        
        Bearish Engulfing:
            - Previous candle is bullish (green)
            - Current candle is bearish (red)
            - Current candle body completely engulfs previous body
            - At resistance level or in downtrend
            - Volume confirmation
        """
        try:
            if df is None or df.empty or len(df) < 20:
                return None

            prev = df.iloc[-2]
            current = df.iloc[-1]
            
            prev_open = float(prev["open"])
            prev_close = float(prev["close"])
            prev_high = float(prev["high"])
            prev_low = float(prev["low"])
            
            curr_open = float(current["open"])
            curr_close = float(current["close"])
            curr_high = float(current["high"])
            curr_low = float(current["low"])
            current_price = curr_close

            # Calculate body sizes
            prev_body = abs(prev_close - prev_open)
            curr_body = abs(curr_close - curr_open)
            
            # Require meaningful candles
            if prev_body < 1 or curr_body < 1:
                return None

            atr = support_resistance.atr
            trend_dir = higher_tf_context.get("trend_direction", "FLAT")
            rsi_15 = float(higher_tf_context.get("rsi_15", 50.0))


            # Volume check (current bar should have higher volume)
            vol_surge, surge_ratio, _ = self._detect_volume_surge(df)

            # =============================
            # BULLISH ENGULFING
            # =============================
            is_bullish_engulfing = (
                prev_close < prev_open and  # Previous bearish
                curr_close > curr_open and  # Current bullish
                curr_open <= prev_close and  # Opens at/below prev close
                curr_close >= prev_open      # Closes at/above prev open
            )

            if is_bullish_engulfing:
                # Check if at support level
                at_support = False
                support_level = None
                
                for level in support_resistance.support_levels[:3]:
                    distance_pct = abs(curr_low - level) / level * 100
                    if distance_pct <= RETEST_ZONE_PERCENT:
                        at_support = True
                        support_level = level
                        break
                
                if not at_support and support_resistance.pdl > 0:
                    distance_pct = abs(curr_low - support_resistance.pdl) / support_resistance.pdl * 100
                    if distance_pct <= RETEST_ZONE_PERCENT:
                        at_support = True
                        support_level = support_resistance.pdl

                # Require either support level OR uptrend
                if at_support or trend_dir == "UP":
                    # Require volume confirmation
                    if not vol_surge:
                        logger.info(f"⏭️ Bullish engulfing skipped: No volume surge ({surge_ratio:.1f}x)")
                        return None
                    
                    logger.info(
                        f"🟢 BULLISH ENGULFING detected | "
                        f"Prev: {prev_open:.2f}->{prev_close:.2f} | "
                        f"Curr: {curr_open:.2f}->{curr_close:.2f} | "
                        f"Vol: {surge_ratio:.1f}x"
                    )

                    entry_price = curr_close
                    stop_loss = curr_low - (atr * 0.5)
                    
                    # Target: Next resistance
                    target = None
                    for r in support_resistance.resistance_levels:
                        if r > current_price:
                            target = r
                            break
                    
                    if not target and support_resistance.pdh > current_price:
                        target = support_resistance.pdh
                    
                    if not target:
                        target = current_price + (atr * 3.0)
                    
                    risk = abs(entry_price - stop_loss)
                    reward = abs(target - entry_price)
                    rr = reward / risk if risk > 0 else 0
                    
                    if rr < 1.5:
                        logger.info(f"⏭️ Bullish engulfing skipped: Poor R:R ({rr:.2f})")
                        return None

                    confidence = 70.0
                    if trend_dir == "UP":
                        confidence += 10
                    if at_support:
                        confidence += 5
                    if rsi_15 < 50:
                        confidence += 5
                    if surge_ratio > 2.0:
                        confidence += 5

                    return Signal(
                        signal_type=SignalType.BULLISH_ENGULFING,
                        instrument=self.instrument,
                        timeframe="5MIN",
                        price_level=support_level if at_support else curr_low,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=target,
                        confidence=min(confidence, 95.0),
                        volume_confirmed=True,
                        momentum_confirmed=(rsi_15 < 50),
                        risk_reward_ratio=rr,
                        timestamp=pd.Timestamp.now(),
                        description=(
                            f"Bullish engulfing pattern | "
                            f"Vol surge: {surge_ratio:.1f}x | RR: {rr:.1f}"
                        ),
                        debug_info={
                            "surge_ratio": surge_ratio,
                            "engulf_ratio": curr_body / prev_body if prev_body > 0 else 0,
                            "trend_dir": trend_dir,
                            "rsi_15": rsi_15,
                        },
                    )

            # =============================
            # BEARISH ENGULFING
            # =============================
            is_bearish_engulfing = (
                prev_close > prev_open and  # Previous bullish
                curr_close < curr_open and  # Current bearish
                curr_open >= prev_close and  # Opens at/above prev close
                curr_close <= prev_open      # Closes at/below prev open
            )

            if is_bearish_engulfing:
                # Check if at resistance level
                at_resistance = False
                resistance_level = None
                
                for level in support_resistance.resistance_levels[:3]:
                    distance_pct = abs(curr_high - level) / level * 100
                    if distance_pct <= RETEST_ZONE_PERCENT:
                        at_resistance = True
                        resistance_level = level
                        break
                
                if not at_resistance and support_resistance.pdh > 0:
                    distance_pct = abs(curr_high - support_resistance.pdh) / support_resistance.pdh * 100
                    if distance_pct <= RETEST_ZONE_PERCENT:
                        at_resistance = True
                        resistance_level = support_resistance.pdh

                # Require either resistance level OR downtrend
                if at_resistance or trend_dir == "DOWN":
                    # Require volume confirmation
                    if not vol_surge:
                        logger.info(f"⏭️ Bearish engulfing skipped: No volume surge ({surge_ratio:.1f}x)")
                        return None
                    
                    logger.info(
                        f"🔴 BEARISH ENGULFING detected | "
                        f"Prev: {prev_open:.2f}->{prev_close:.2f} | "
                        f"Curr: {curr_open:.2f}->{curr_close:.2f} | "
                        f"Vol: {surge_ratio:.1f}x"
                    )

                    entry_price = curr_close
                    stop_loss = curr_high + (atr * 0.5)
                    
                    # Target: Next support
                    target = None
                    for s in support_resistance.support_levels:
                        if s < current_price:
                            target = s
                            break
                    
                    if not target and support_resistance.pdl > 0 and support_resistance.pdl < current_price:
                        target = support_resistance.pdl
                    
                    if not target:
                        target = current_price - (atr * 3.0)
                    
                    risk = abs(stop_loss - entry_price)
                    reward = abs(entry_price - target)
                    rr = reward / risk if risk > 0 else 0
                    
                    if rr < 1.5:
                        logger.info(f"⏭️ Bearish engulfing skipped: Poor R:R ({rr:.2f})")
                        return None

                    confidence = 70.0
                    if trend_dir == "DOWN":
                        confidence += 10
                    if at_resistance:
                        confidence += 5
                    if rsi_15 > 50:
                        confidence += 5
                    if surge_ratio > 2.0:
                        confidence += 5

                    return Signal(
                        signal_type=SignalType.BEARISH_ENGULFING,
                        instrument=self.instrument,
                        timeframe="5MIN",
                        price_level=resistance_level if at_resistance else curr_high,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=target,
                        confidence=min(confidence, 95.0),
                        volume_confirmed=True,
                        momentum_confirmed=(rsi_15 > 50),
                        risk_reward_ratio=rr,
                        timestamp=pd.Timestamp.now(),
                        description=(
                            f"Bearish engulfing pattern | "
                            f"Vol surge: {surge_ratio:.1f}x | RR: {rr:.1f}"
                        ),
                        debug_info={
                            "surge_ratio": surge_ratio,
                            "engulf_ratio": curr_body / prev_body if prev_body > 0 else 0,
                            "trend_dir": trend_dir,
                            "rsi_15": rsi_15,
                        },
                    )

            return None

        except Exception as e:
            logger.error(f"❌ Engulfing detection failed: {str(e)}")
            return None

    def _is_choppy_session(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Detect if market is choppy/ranging (avoid trading in such conditions).
        
        Returns:
            (is_choppy, reason)
        
        Criteria:
            - ATR < 0.3% of price (very low volatility)
            - Price oscillating around VWAP (4+ crosses in 10 bars)
        """
        try:
            from config.settings import MIN_ATR_PERCENT, MAX_VWAP_CROSSES
            
            if df is None or len(df) < 20:
                return False, ""
            
            current_price = float(df.iloc[-1]["close"])
            atr = self._calculate_atr(df)
            
            # 1. Check ATR (volatility)
            atr_pct = (atr / current_price) * 100
            
            if atr_pct < MIN_ATR_PERCENT:
                return True, f"Low volatility (ATR: {atr_pct:.2f}%)"
            else:
                logger.info(f"✅ Volatility OK | ATR: {atr_pct:.4f}% (> {MIN_ATR_PERCENT}%)")
            
            # 2. Check VWAP oscillation (choppy price action)
            _, vwap, _ = self._calculate_vwap(df)
            recent_closes = df.tail(10)["close"]
            
            # Count VWAP crosses
            crosses = 0
            for i in range(1, len(recent_closes)):
                prev_close = recent_closes.iloc[i-1]
                curr_close = recent_closes.iloc[i]
                
                if (prev_close < vwap and curr_close > vwap) or \
                   (prev_close > vwap and curr_close < vwap):
                    crosses += 1
            
            if crosses >= MAX_VWAP_CROSSES:
                return True, f"Choppy (VWAP crosses: {crosses})"
            
            return False, ""
            
        except Exception as e:
            logger.warning(f"⚠️ Choppy session check failed: {str(e)}")
            return False, ""
    # =====================================================================
    # TOP-LEVEL ANALYSIS WRAPPER (WITH MTF)
    # =====================================================================

    def analyze_with_multi_tf(
        self, df_5m: pd.DataFrame, higher_tf_context: Dict, df_15m: pd.DataFrame = None
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
            "pin_bar_signal": None,
            "engulfing_signal": None,
        }

        try:
            if df_5m is None or df_5m.empty:
                logger.error("❌ analyze_with_multi_tf: empty 5m data")
                return result

            pdh, pdl = self.calculate_pdh_pdl(df_5m)
            
            # Use 15m data for Support/Resistance if available (Clean Levels)
            # Otherwise fall back to 5m (Noisy but better than nothing)
            sr_source_df = df_15m if df_15m is not None and not df_15m.empty else df_5m
            levels = self.calculate_support_resistance(sr_source_df)
            
            vol_confirmed, vol_ratio, _ = self.check_volume_confirmation(df_5m)
            breakout = self.detect_breakout(df_5m, levels, higher_tf_context)
            retest = self.detect_retest_setup(df_5m, levels, higher_tf_context)
            inside_bar = self.detect_inside_bar(df_5m, higher_tf_context, levels)
            pin_bar = self.detect_pin_bar(df_5m, levels, higher_tf_context)
            engulfing = self.detect_engulfing(df_5m, levels, higher_tf_context)

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
                    "pin_bar_signal": pin_bar,
                    "engulfing_signal": engulfing,
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

```

## ML Module

### ./ml_module/__init__.py

```python
"""
ML Module for LightGBM-based Signal Quality Prediction
Optimized for Google Cloud Platform
"""

__version__ = "1.0.0"

from ml_module.feature_extractor import extract_features
from ml_module.predictor import SignalQualityPredictor
from ml_module.model_storage import ModelStorage

__all__ = [
    "extract_features",
    "SignalQualityPredictor", 
    "ModelStorage"
]

```

### ./ml_module/feature_extractor.py

```python
"""
Feature Extraction for LightGBM Model
Converts signal + context data into ML features
"""

import logging
from typing import Dict
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)


def extract_features(
    signal: Dict,
    technical_context: Dict,
    option_metrics: Dict,
    market_status: Dict = None
) -> Dict:
    """
    Extract 30+ features from signal data for ML prediction.
    
    Args:
        signal: Trading signal with type, entry, SL, TP, etc.
        technical_context: MTF analysis, trends, indicators
        option_metrics: PCR, IV, OI data
        market_status: Optional choppy/volatile conditions
        
    Returns:
        Dict of features for LightGBM
    """
    try:
        # Initialize features dict
        features = {}
        
        # =================================================================
        # 1. SIGNAL CHARACTERISTICS
        # =================================================================
        features["signal_type"] = signal.get("signal_type", "UNKNOWN")  # Categorical
        features["confidence"] = float(signal.get("confidence", 0))
        features["volume_confirmed"] = int(signal.get("volume_confirmed", False))
        features["momentum_confirmed"] = int(signal.get("momentum_confirmed", False))
        features["risk_reward"] = float(signal.get("risk_reward_ratio", 0))
        
        # Price levels
        entry = float(signal.get("entry_price", 0))
        sl = float(signal.get("stop_loss", 0))
        tp = float(signal.get("take_profit", 0))
        
        features["stop_loss_pct"] = abs(entry - sl) / entry * 100 if entry > 0 else 0
        features["target_pct"] = abs(tp - entry) / entry * 100 if entry > 0 else 0
        
        # =================================================================
        # 2. MULTI-TIMEFRAME CONTEXT
        # =================================================================
        htf_context = technical_context.get("higher_tf_context", {})
        
        features["trend_5m"] = htf_context.get("trend_5m", "NEUTRAL")  # Categorical
        features["trend_15m"] = htf_context.get("trend_15m", "NEUTRAL")
        features["trend_daily"] = htf_context.get("trend_daily", "NEUTRAL")
        
        # Trend alignment (binary)
        signal_direction = "UP" if "BULLISH" in features["signal_type"] or "SUPPORT" in features["signal_type"] else "DOWN"
        features["trend_aligned_15m"] = int(
            (signal_direction == "UP" and features["trend_15m"] == "UP") or
            (signal_direction == "DOWN" and features["trend_15m"] == "DOWN")
        )
        features["trend_aligned_daily"] = int(
            (signal_direction == "UP" and features["trend_daily"] == "UP") or
            (signal_direction == "DOWN" and features["trend_daily"] == "DOWN")
        )
        
        # =================================================================
        # 3. PRICE STRUCTURE INDICATORS
        # =================================================================
        vwap = htf_context.get("vwap_5m", entry)
        ema20 = htf_context.get("ema20", entry)
        ema50 = htf_context.get("ema50", entry)
        
        features["distance_to_vwap_pct"] = (entry - vwap) / vwap * 100 if vwap > 0 else 0
        features["distance_to_ema20_pct"] = (entry - ema20) / ema20 * 100 if ema20 > 0 else 0
        features["distance_to_ema50_pct"] = (entry - ema50) / ema50 * 100 if ema50 > 0 else 0
        
        features["above_vwap"] = int(entry > vwap)
        features["above_ema20"] = int(entry > ema20)
        
        # ATR-based volatility
        features["atr_percent"] = float(htf_context.get("atr_percent", 0))
        
        # =================================================================
        # 4. OPTIONS DATA
        # =================================================================
        features["pcr"] = float(option_metrics.get("pcr", 1.0))
        features["iv"] = float(option_metrics.get("iv", 15))
        
        oi_data = option_metrics.get("oi_change", {})
        features["oi_sentiment"] = oi_data.get("sentiment", "NEUTRAL")  # Categorical
        
        # Option alignment with signal
        oi_bullish = features["oi_sentiment"] == "BULLISH"
        oi_bearish = features["oi_sentiment"] == "BEARISH"
        features["oi_aligned"] = int(
            (signal_direction == "UP" and oi_bullish) or
            (signal_direction == "DOWN" and oi_bearish)
        )
        
        # =================================================================
        # 5. TIME-BASED FEATURES
        # =================================================================
        ist = pytz.timezone("Asia/Kolkata")
        now = datetime.now(ist)
        
        features["hour_of_day"] = now.hour
        features["minute_of_hour"] = now.minute
        features["day_of_week"] = now.weekday()  # 0=Monday, 4=Friday
        
        # Market session
        market_open = datetime.strptime("09:15", "%H:%M").time()
        current_time = now.time()
        minutes_from_open = (now.hour * 60 + now.minute) - (9 * 60 + 15)
        
        features["minutes_from_open"] = max(0, minutes_from_open)
        features["is_first_hour"] = int(minutes_from_open < 60)
        features["is_last_hour"] = int(now.hour >= 14 and now.minute >= 30)
        features["is_lunch_hour"] = int(12 <= now.hour < 13)
        
        # =================================================================
        # 6. MARKET CONDITIONS
        # =================================================================
        features["india_vix"] = float(htf_context.get("india_vix", 15))
        features["vix_regime"] = "HIGH" if features["india_vix"] > 20 else "NORMAL"  # Categorical
        
        if market_status:
            features["is_choppy"] = int(market_status.get("is_choppy", False))
        else:
            features["is_choppy"] = 0
        
        # =================================================================
        # 7. PATTERN-SPECIFIC FEATURES
        # =================================================================
        is_breakout = "BREAKOUT" in features["signal_type"]
        is_retest = "RETEST" in features["signal_type"] or "BOUNCE" in features["signal_type"]
        is_reversal = "PIN_BAR" in features["signal_type"] or "ENGULFING" in features["signal_type"]
        
        features["is_breakout"] = int(is_breakout)
        features["is_retest"] = int(is_retest)
        features["is_reversal"] = int(is_reversal)
        features["is_inside_bar"] = int("INSIDE_BAR" in features["signal_type"])
        
        logger.debug(f"Extracted {len(features)} features for {features['signal_type']}")
        
        return features
        
    except Exception as e:
        logger.error(f"❌ Feature extraction failed: {e}", exc_info=True)
        # Return minimal features on error
        return {
            "signal_type": "ERROR",
            "confidence": 0,
            "risk_reward": 0
        }


def get_feature_names() -> list:
    """Return list of all feature names for model training."""
    return [
        # Signal characteristics
        "signal_type", "confidence", "volume_confirmed", "momentum_confirmed",
        "risk_reward", "stop_loss_pct", "target_pct",
        
        # Multi-timeframe
        "trend_5m", "trend_15m", "trend_daily",
        "trend_aligned_15m", "trend_aligned_daily",
        
        # Price structure
        "distance_to_vwap_pct", "distance_to_ema20_pct", "distance_to_ema50_pct",
        "above_vwap", "above_ema20", "atr_percent",
        
        # Options
        "pcr", "iv", "oi_sentiment", "oi_aligned",
        
        # Time-based
        "hour_of_day", "minute_of_hour", "day_of_week",
        "minutes_from_open", "is_first_hour", "is_last_hour", "is_lunch_hour",
        
        # Market conditions
        "india_vix", "vix_regime", "is_choppy",
        
        # Pattern types
        "is_breakout", "is_retest", "is_reversal", "is_inside_bar"
    ]


def get_categorical_features() -> list:
    """Return list of categorical feature names."""
    return [
        "signal_type",
        "trend_5m",
        "trend_15m",
        "trend_daily",
        "oi_sentiment",
        "vix_regime"
    ]

```

### ./ml_module/model_storage.py

```python
"""
Model Storage Handler for Google Cloud Storage
Manages loading and saving LightGBM models from/to GCS
"""

import os
import logging
import tempfile
from typing import Optional
from google.cloud import storage
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelStorage:
    """Handle model storage and retrieval from Google Cloud Storage."""
    
    def __init__(self, bucket_name: str, model_name: str = "signal_quality_v1.txt"):
        """
        Initialize GCS model storage.
        
        Args:
            bucket_name: GCS bucket name (without gs:// prefix)
            model_name: Model filename in bucket
        """
        self.bucket_name = bucket_name
        self.model_name = model_name
        self.local_cache_dir = "/tmp/ml_models"  # Cloud Functions writable directory
        
        # Create cache directory
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(bucket_name)
            logger.info(f"✅ Connected to GCS bucket: {bucket_name}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to GCS: {e}")
            self.bucket = None
    
    def download_model(self, force_refresh: bool = False) -> Optional[str]:
        """
        Download model from GCS to local cache.
        
        Args:
            force_refresh: Force download even if cached
            
        Returns:
            Local path to model file, or None if failed
        """
        if not self.bucket:
            logger.error("GCS bucket not available")
            return None
        
        local_path = os.path.join(self.local_cache_dir, self.model_name)
        
        # Check cache
        if os.path.exists(local_path) and not force_refresh:
            logger.debug(f"Using cached model: {local_path}")
            return local_path
        
        try:
            blob = self.bucket.blob(f"models/{self.model_name}")
            
            if not blob.exists():
                logger.error(f"Model not found in GCS: models/{self.model_name}")
                return None
            
            # Download
            blob.download_to_filename(local_path)
            logger.info(f"✅ Downloaded model from GCS: {self.model_name}")
            
            return local_path
            
        except Exception as e:
            logger.error(f"❌ Failed to download model from GCS: {e}")
            return None
    
    def upload_model(self, local_model_path: str, version: Optional[str] = None) -> bool:
        """
        Upload trained model to GCS.
        
        Args:
            local_model_path: Path to local model file
            version: Optional version string (default: timestamp)
            
        Returns:
            True if successful
        """
        if not self.bucket:
            logger.error("GCS bucket not available")
            return False
        
        try:
            # Create versioned filename
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            versioned_name = f"signal_quality_{version}.txt"
            
            # Upload versioned file
            blob_versioned = self.bucket.blob(f"models/versions/{versioned_name}")
            blob_versioned.upload_from_filename(local_model_path)
            logger.info(f"✅ Uploaded versioned model: {versioned_name}")
            
            # Update active model (symlink equivalent)
            blob_active = self.bucket.blob(f"models/{self.model_name}")
            blob_active.upload_from_filename(local_model_path)
            logger.info(f"✅ Updated active model: {self.model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload model to GCS: {e}")
            return False
    
    def list_model_versions(self, limit: int = 10) -> list:
        """
        List available model versions in GCS.
        
        Args:
            limit: Max number of versions to return
            
        Returns:
            List of model version filenames
        """
        if not self.bucket:
            return []
        
        try:
            blobs = self.bucket.list_blobs(prefix="models/versions/", max_results=limit)
            versions = [blob.name.split("/")[-1] for blob in blobs]
            return sorted(versions, reverse=True)  # Newest first
            
        except Exception as e:
            logger.error(f"❌ Failed to list model versions: {e}")
            return []

```

### ./ml_module/predictor.py

```python
"""
LightGBM Signal Quality Predictor
Optimized for Google Cloud Functions
"""

import logging
import lightgbm as lgb
from typing import Dict, Optional
import pandas as pd

from ml_module.model_storage import ModelStorage
from ml_module.feature_extractor import get_feature_names, get_categorical_features

logger = logging.getLogger(__name__)


class SignalQualityPredictor:
    """Predict signal quality using LightGBM model."""
    
    def __init__(self, bucket_name: str, model_name: str = "signal_quality_v1.txt"):
        """
        Initialize predictor with GCS model.
        
        Args:
            bucket_name: GCS bucket for models
            model_name: Model filename
        """
        self.model = None
        self.enabled = False
        self.feature_names = get_feature_names()
        self.categorical_features = get_categorical_features()
        
        # Initialize model storage
        self.storage = ModelStorage(bucket_name, model_name)
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> bool:
        """
        Load LightGBM model from GCS.
        
        Returns:
            True if successful
        """
        try:
            # Download model from GCS
            model_path = self.storage.download_model()
            
            if model_path is None:
                logger.warning("⚠️ Model file not available, ML filtering disabled")
                return False
            
            # Load LightGBM model
            self.model = lgb.Booster(model_file=model_path)
            self.enabled = True
            
            logger.info(f"✅ LightGBM Model Loaded | Features: {len(self.feature_names)}")
            return True
            
        except FileNotFoundError:
            logger.warning("⚠️ Model file not found, ML filtering disabled")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def predict_quality(self, features: Dict) -> Optional[float]:
        """
        Predict signal win probability.
        
        Args:
            features: Feature dict from feature_extractor
            
        Returns:
            Probability [0-1], or None if model disabled/error
        """
        if not self.enabled or self.model is None:
            return None
        
        try:
            # Convert features dict to DataFrame
            # LightGBM expects 2D array (even for single prediction)
            df = pd.DataFrame([features])
            
            # Ensure all expected features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    logger.warning(f"Missing feature: {feature}, using default")
                    df[feature] = 0 if feature not in self.categorical_features else "UNKNOWN"
            
            # Reorder columns to match training
            df = df[self.feature_names]
            
            # Predict (returns array)
            prediction = self.model.predict(df)[0]
            
            logger.debug(f"ML Prediction: {prediction:.3f}")
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}", exc_info=True)
            return None
    
    def predict_with_threshold(
        self,
        features: Dict,
        threshold: float = 0.65
    ) -> tuple[bool, float]:
        """
        Predict and return both decision and probability.
        
        Args:
            features: Feature dict
            threshold: Minimum probability to accept signal
            
        Returns:
            (should_accept, probability)
        """
        prob = self.predict_quality(features)
        
        if prob is None:
            # Fallback: accept signal if model unavailable
            return True, 0.5
        
        should_accept = prob >= threshold
        return should_accept, prob
    
    def reload_model(self) -> bool:
        """
        Force reload model from GCS (for updates).
        
        Returns:
            True if successful
        """
        logger.info("🔄 Reloading model from GCS...")
        return self._load_model()
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get feature importance from model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dict of {feature_name: importance_score}
        """
        if not self.enabled or self.model is None:
            return {}
        
        try:
            importance = self.model.feature_importance()
            feature_importance = dict(zip(self.feature_names, importance))
            
            # Sort and return top N
            sorted_importance = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return dict(sorted_importance[:top_n])
            
        except Exception as e:
            logger.error(f"❌ Failed to get feature importance: {e}")
            return {}

```

## AI Module

### ./ai_module/ai_factory.py

```python
"""
AI Factory Module
Provides abstraction layer for AI analyzer selection (Groq vs Vertex AI vs Hybrid).
"""

import logging
import random
from typing import Dict, Optional

from config.settings import (
    AI_PROVIDER,
    HYBRID_GROQ_WEIGHT,
    HYBRID_VERTEX_WEIGHT,
)

logger = logging.getLogger(__name__)


class HybridAnalyzer:
    """
    Hybrid AI analyzer that routes requests between Groq and Vertex AI.
    Useful for A/B testing to compare quality and performance.
    """
    
    def __init__(self):
        from ai_module.groq_analyzer import get_analyzer as get_groq
        from ai_module.vertex_analyzer import get_analyzer as get_vertex
        
        self.groq = get_groq()
        self.vertex = get_vertex()
        self.groq_weight = HYBRID_GROQ_WEIGHT
        self.vertex_weight = HYBRID_VERTEX_WEIGHT
        
        # Normalize weights
        total = self.groq_weight + self.vertex_weight
        self.groq_weight /= total
        self.vertex_weight /= total
        
        logger.info(
            f"🔀 Hybrid AI Mode | Groq: {self.groq_weight*100:.0f}% | "
            f"Vertex: {self.vertex_weight*100:.0f}%"
        )
    
    def analyze_signal(
        self,
        signal_data: Dict,
        market_context: Dict,
        technical_data: Dict
    ) -> Optional[Dict]:
        """Route to Groq or Vertex based on configured weights."""
        
        # Random selection based on weights
        if random.random() < self.groq_weight:
            logger.debug("🎲 Hybrid: Using Groq")
            result = self.groq.analyze_signal(signal_data, market_context, technical_data)
            if result:
                result["ai_provider"] = "GROQ"
            return result
        else:
            logger.debug("🎲 Hybrid: Attempting Vertex AI")
            try:
                result = self.vertex.analyze_signal(signal_data, market_context, technical_data)
                if result and "error" not in result:
                    result["ai_provider"] = "VERTEX"
                    return result
                else:
                    logger.warning(f"⚠️ Vertex AI returned error or empty result, falling back to Groq")
            except Exception as e:
                logger.error(f"❌ Vertex AI failed in Hybrid mode: {e}. Falling back to Groq.")
            
            # Fallback to Groq
            result = self.groq.analyze_signal(signal_data, market_context, technical_data)
            if result:
                result["ai_provider"] = "GROQ (Fallback)"
            return result
    
    def test_connection(self) -> bool:
        """Test both providers."""
        groq_ok = self.groq.test_connection()
        vertex_ok = self.vertex.test_connection()
        
        if not groq_ok and not vertex_ok:
            return False
        
        if not groq_ok:
            logger.warning("⚠️ Groq unavailable in hybrid mode - using Vertex only")
        if not vertex_ok:
            logger.warning("⚠️ Vertex unavailable in hybrid mode - using Groq only")
        
        return True
    
    def get_usage_stats(self) -> Dict:
        """Return combined stats from both providers."""
        return {
            "mode": "HYBRID",
            "groq_weight": f"{self.groq_weight*100:.0f}%",
            "vertex_weight": f"{self.vertex_weight*100:.0f}%",
            "groq_stats": self.groq.get_usage_stats(),
            "vertex_stats": self.vertex.get_usage_stats(),
        }


def get_analyzer(provider: str = None):
    """
    Factory function to get appropriate AI analyzer.
    
    Args:
        provider: AI provider to use (GROQ, VERTEX, HYBRID). 
                  If None, uses AI_PROVIDER from settings.
    
    Returns:
        AI analyzer instance with analyze_signal() method.
    
    Example:
        analyzer = get_analyzer()
        result = analyzer.analyze_signal(signal_data, market_context, technical_data)
    """
    provider = (provider or AI_PROVIDER).upper()
    
    if provider == "GROQ":
        logger.debug("🧠 Using Groq AI")
        from ai_module.groq_analyzer import get_analyzer as get_groq
        return get_groq()
    
    elif provider == "VERTEX":
        logger.debug("🧠 Using Vertex AI (Gemini)")
        from ai_module.vertex_analyzer import get_analyzer as get_vertex
        return get_vertex()
    
    elif provider == "HYBRID":
        logger.debug("🧠 Using Hybrid AI Mode")
        return HybridAnalyzer()
    
    else:
        logger.error(f"❌ Unknown AI provider: {provider}. Falling back to Groq.")
        from ai_module.groq_analyzer import get_analyzer as get_groq
        return get_groq()


# Singleton instance
_analyzer_instance = None


def get_default_analyzer():
    """
    Get singleton analyzer instance using configured provider.
    Cached for performance.
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = get_analyzer()
    return _analyzer_instance

```

### ./ai_module/groq_analyzer.py

```python
"""
Groq AI Analyzer Module
Uses LLaMA 3 70B via Groq API to provide "Hedge Fund Analyst" reasoning for technical signals.
"""

import os
import json
import logging
import requests
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from config.settings import (
    GROQ_API_KEY, 
    GROQ_MODEL, 
    GROQ_TEMPERATURE, 
    GROQ_MAX_TOKENS
)

logger = logging.getLogger(__name__)

class GroqAnalyzer:
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = GROQ_MODEL
        
        if not self.api_key or "YOUR_KEY" in self.api_key:
            logger.warning("⚠️ Groq API Key not found. AI Analysis disabled.")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"🧠 Groq AI Initialized | Model: {self.model}")

        # Setup resilient session
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def analyze_signal(
        self, 
        signal_data: Dict, 
        market_context: Dict, 
        technical_data: Dict
    ) -> Dict:
        """
        Analyze a trading signal using LLaMA 3.
        
        Returns:
            Dict containing:
            - reasoning (str): Natural language explanation
            - confidence (int): 0-100 score
            - risks (List[str]): Potential risks
            - verdict (str): "STRONG BUY", "CAUTIOUS BUY", "PASS"
        """
        if not self.enabled:
            return {"error": "AI Disabled"}

        try:
            prompt = self._construct_prompt(signal_data, market_context, technical_data)
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                "temperature": GROQ_TEMPERATURE,
                "max_tokens": GROQ_MAX_TOKENS,
                "response_format": {"type": "json_object"}
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            start_time = datetime.now()
            response = self.session.post(
                self.api_url, 
                json=payload, 
                headers=headers, 
                timeout=10
            ) 
            response.raise_for_status()
            
            latency = (datetime.now() - start_time).total_seconds()
            result = response.json()
            
            ai_content = result['choices'][0]['message']['content']
            parsed_result = json.loads(ai_content)
            
            logger.info(f"🤖 AI Analysis Complete ({latency:.2f}s) | Confidence: {parsed_result.get('confidence')}%")
            
            return parsed_result

        except Exception as e:
            logger.error(f"❌ Groq Analysis Failed: {str(e)}")
            return None

    def test_connection(self) -> bool:
        """Test if Groq API is reachable and key is valid."""
        if not self.enabled:
            return False
            
        try:
            # Lightweight test call (list models)
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = self.session.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"⚠️ Groq Connection Test Failed: {e}")
            return False

    def get_usage_stats(self) -> Dict:
        """Return usage statistics for the AI module."""
        return {
            "enabled": self.enabled,
            "model": self.model,
            "requests_today": 0,  # Not implemented - usage tracked at Groq API level
            "tokens_used": 0
        }

    def _get_system_prompt(self) -> str:
        return """You are a Senior Hedge Fund Technical Analyst. 
Your job is to VALIDATE algorithmic trading signals for NIFTY 50.
You are skeptical, risk-averse, and focus on CONFLUENCE.

OUTPUT FORMAT (JSON):
{
    "verdict": "STRONG_BUY" | "CAUTIOUS_BUY" | "STRONG_SELL" | "CAUTIOUS_SELL" | "PASS",
    "confidence": <0-100 integer>,
    "reasoning": "<Concise 2-sentence explanation focusing on WHY this works or fails>",
    "risks": ["<Risk 1>", "<Risk 2>"]
}

SCORING RULES:
- High Confidence (>80): Needs Trend Alignment + Structure Breakout + Good R:R.
- Medium Confidence (60-80): Good structure but mixed trend or low volume.
- Fail (<50): Counter-trend without reversal structure, or poor metrics.
- DIRECTION: Ensure verdict matches signal direction (SELL for Bearish, BUY for Bullish)."""

    def forecast_market_outlook(self, daily_summary_text: str) -> Dict:
        """
        Generate a market outlook forecast based on the day's summary.
        
        Args:
            daily_summary_text (str): A text summary of the day's price action and signals.
            
        Returns:
            Dict: {
                "outlook": "BULLISH" | "BEARISH" | "NEUTRAL",
                "confidence": int,
                "summary": str
            }
        """
        if not self.enabled:
            return {"outlook": "NEUTRAL", "confidence": 0, "summary": "AI Disabled"}

        try:
            system_prompt = """You are a Market Strategist. 
Analyze the provided end-of-day market summary and forecast the outlook for TOMORROW.
Consider trend, support/resistance tests, and overall sentiment.
OUTPUT JSON: {"outlook": "BULLISH"|"BEARISH"|"NEUTRAL", "confidence": 0-100, "summary": "One sentence outlook."}"""

            user_prompt = f"MARKET SUMMARY:\n{daily_summary_text}\n\nFORECAST THE NEXT SESSION:"

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 150,
                "response_format": {"type": "json_object"}
            }

            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = self.session.post(
                self.api_url, 
                json=payload, 
                headers=headers, 
                timeout=10
            ) 
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            return json.loads(content)

        except Exception as e:
            logger.error(f"❌ Forecast failed: {e}")
            return {"outlook": "NEUTRAL", "confidence": 0, "summary": "Forecast Error"}


    def _construct_prompt(self, sig: Dict, context: Dict, tech: Dict) -> str:
        """Construct the dynamic prompt with signal details."""
        
        # safely access keys
        signal_type = sig.get('signal_type', 'UNKNOWN')
        price = sig.get('price_level', 0)
        trend_15m = context.get('trend_direction', 'FLAT')
        entry_price = float(sig.get('entry_price', 0))
        stop_loss = float(sig.get('stop_loss', 0))
        
        # Determine signal direction from entry vs stop loss
        # If SL > Entry = SHORT, if SL < Entry = LONG
        if stop_loss > entry_price:
            direction = "SHORT"
        elif stop_loss < entry_price:
            direction = "LONG"
        else:
            # Fallback: try to infer from signal type keywords
            if any(x in signal_type.upper() for x in ["BEARISH", "RESISTANCE", "SHORT", "BREAKDOWN"]):
                direction = "SHORT"
            else:
                direction = "LONG"
        
        mtf_data = (
            f"15m Trend: {trend_15m}\n"
            f"Rel to VWAP: {'Above' if context.get('price_above_vwap') else 'Below'}\n"
            f"Rel to EMA20: {'Above' if context.get('price_above_ema20') else 'Below'}"
        )
        
        option_data = "N/A"
        # If we have option metrics passed in "tech" or "context"
        # Adapted based on usage in main.py
        
        return f"""
ANALYZE THIS TRADE SETUP:

INSTRUMENT: NIFTY 50
SIGNAL: {signal_type}
DIRECTION: {direction}  ← CRITICAL: Use SELL verdicts for SHORT, BUY verdicts for LONG
LEVEL: {price}
ENTRY: {entry_price}
STOP LOSS: {stop_loss}

TECHNICAL CONTEXT:
{mtf_data}

SIGNAL METRICS:
- Confidence: {sig.get('confidence')}%
- R:R Ratio: {sig.get('risk_reward_ratio', 0):.2f}
- Description: {sig.get('description')}

NOTE: This is an INDEX instrument - volume data is not available/relevant.

CRITICAL: Your verdict MUST match the DIRECTION above:
- If DIRECTION is SHORT → use STRONG_SELL or CAUTIOUS_SELL (NOT BUY)
- If DIRECTION is LONG → use STRONG_BUY or CAUTIOUS_BUY (NOT SELL)

Evaluate based on Multi-Timeframe alignment and Market Structure.
"""

# Singleton access
_analyzer_instance = None

def get_analyzer():
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = GroqAnalyzer()
    return _analyzer_instance

```

### ./ai_module/vertex_analyzer.py

```python
"""
Vertex AI Gemini Analyzer Module
Uses Gemini 1.5 Pro via Vertex AI for trading signal analysis.
Drop-in replacement for Groq with enhanced capabilities.
"""

import json
import logging
from typing import Dict, Optional
from datetime import datetime

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    logging.warning("⚠️ Vertex AI SDK not installed. Run: pip install google-cloud-aiplatform")

from config.settings import (
    VERTEX_PROJECT_ID,
    VERTEX_LOCATION,
    VERTEX_MODEL,
)

logger = logging.getLogger(__name__)


class VertexAnalyzer:
    """Gemini-based signal analyzer using Vertex AI."""
    
    def __init__(self):
        self.project_id = VERTEX_PROJECT_ID
        self.location = VERTEX_LOCATION
        self.model_name = VERTEX_MODEL
        
        if not VERTEX_AVAILABLE:
            logger.error("❌ Vertex AI SDK not available")
            self.enabled = False
            return
        
        if not self.project_id or "YOUR_PROJECT" in self.project_id:
            logger.warning("⚠️ Vertex AI Project ID not configured. AI disabled.")
            self.enabled = False
            return
        
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location=self.location)
            
            # Initialize Gemini model
            self.model = GenerativeModel(
                self.model_name,
                generation_config=GenerationConfig(
                    temperature=0.3,  # Lower than Groq for more consistent analysis
                    max_output_tokens=500,
                    response_mime_type="application/json",  # Force JSON output
                )
            )
            
            self.enabled = True
            logger.info(f"🧠 Vertex AI Gemini Initialized | Model: {self.model_name} | Project: {self.project_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Vertex AI: {str(e)}")
            self.enabled = False
    
    def analyze_signal(
        self,
        signal_data: Dict,
        market_context: Dict,
        technical_data: Dict
    ) -> Optional[Dict]:
        """
        Analyze a trading signal using Gemini 1.5 Pro.
        
        Returns:
            Dict containing:
            - reasoning (str): Natural language explanation
            - confidence (int): 0-100 score
            - risks (List[str]): Potential risks
            - verdict (str): "STRONG_BUY", "CAUTIOUS_BUY", "STRONG_SELL", "CAUTIOUS_SELL", "PASS"
        """
        if not self.enabled:
            return {"error": "Vertex AI Disabled"}
        
        try:
            # Construct prompt using same logic as Groq
            system_prompt = self._get_system_prompt()
            user_prompt = self._construct_prompt(signal_data, market_context, technical_data)
            
            # Combine system and user prompts (Gemini doesn't have separate system role)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            start_time = datetime.now()
            
            # Generate content
            response = self.model.generate_content(full_prompt)
            
            latency = (datetime.now() - start_time).total_seconds()
            
            # Parse JSON response
            result_text = response.text.strip()
            
            # Try to extract JSON if wrapped in markdown
            if result_text.startswith("```json"):
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif result_text.startswith("```"):
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            parsed_result = json.loads(result_text)
            
            logger.info(
                f"🤖 Vertex AI Analysis Complete ({latency:.2f}s) | "
                f"Verdict: {parsed_result.get('verdict')} | "
                f"Confidence: {parsed_result.get('confidence')}%"
            )
            
            return parsed_result
        
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse Vertex AI JSON response: {str(e)}")
            logger.error(f"   Raw response: {response.text[:200]}...")
            return None
        
        except Exception as e:
            logger.error(f"❌ Vertex AI Analysis Failed: {str(e)}")
            return None
    
    def test_connection(self) -> bool:
        """Test if Vertex AI is accessible and configured correctly."""
        if not self.enabled:
            return False
        
        try:
            # Simple test prompt
            test_prompt = "Respond with valid JSON: {\"status\": \"ok\"}"
            response = self.model.generate_content(test_prompt)
            
            # Check if we got a response
            if response and response.text:
                logger.info("✅ Vertex AI connection test successful")
                return True
            
            return False
        
        except Exception as e:
            logger.warning(f"⚠️ Vertex AI Connection Test Failed: {e}")
            return False
    
    def get_usage_stats(self) -> Dict:
        """Return usage statistics for the AI module."""
        return {
            "enabled": self.enabled,
            "provider": "Vertex AI",
            "model": self.model_name,
            "project": self.project_id,
            "location": self.location,
            "requests_today": 0,  # Not tracked locally - available in GCP console
        }
    
    def _get_system_prompt(self) -> str:
        """Same system prompt as Groq for consistency."""
        return """You are a Senior Hedge Fund Technical Analyst. 
Your job is to VALIDATE algorithmic trading signals for NIFTY 50.
You are skeptical, risk-averse, and focus on CONFLUENCE.

OUTPUT FORMAT (JSON):
{
    "verdict": "STRONG_BUY" | "CAUTIOUS_BUY" | "STRONG_SELL" | "CAUTIOUS_SELL" | "PASS",
    "confidence": <0-100 integer>,
    "reasoning": "<Concise 2-sentence explanation focusing on WHY this works or fails>",
    "risks": ["<Risk 1>", "<Risk 2>"]
}

SCORING RULES:
- High Confidence (>80): Needs Trend Alignment + Structure Breakout + Good R:R.
- Medium Confidence (60-80): Good structure but mixed trend.
- Fail (<50): Counter-trend without reversal structure, or poor metrics.
- DIRECTION: Ensure verdict matches signal direction (SELL for SHORT, BUY for LONG)."""
    
    def _construct_prompt(self, sig: Dict, context: Dict, tech: Dict) -> str:
        """Construct the dynamic prompt with signal details."""
        
        # Extract signal info
        signal_type = sig.get('signal_type', 'UNKNOWN')
        price = sig.get('price_level', 0)
        trend_15m = context.get('trend_direction', 'FLAT')
        entry_price = float(sig.get('entry_price', 0))
        stop_loss = float(sig.get('stop_loss', 0))
        
        # Determine signal direction from entry vs stop loss
        if stop_loss > entry_price:
            direction = "SHORT"
        elif stop_loss < entry_price:
            direction = "LONG"
        else:
            # Fallback: infer from signal type
            if any(x in signal_type.upper() for x in ["BEARISH", "RESISTANCE", "SHORT", "BREAKDOWN"]):
                direction = "SHORT"
            else:
                direction = "LONG"
        
        mtf_data = (
            f"15m Trend: {trend_15m}\n"
            f"Rel to VWAP: {'Above' if context.get('price_above_vwap') else 'Below'}\n"
            f"Rel to EMA20: {'Above' if context.get('price_above_ema20') else 'Below'}"
        )
        
        return f"""
ANALYZE THIS TRADE SETUP:

INSTRUMENT: NIFTY 50
SIGNAL: {signal_type}
DIRECTION: {direction}  ← CRITICAL: Use SELL verdicts for SHORT, BUY verdicts for LONG
LEVEL: {price}
ENTRY: {entry_price}
STOP LOSS: {stop_loss}

TECHNICAL CONTEXT:
{mtf_data}

SIGNAL METRICS:
- Confidence: {sig.get('confidence')}%
- R:R Ratio: {sig.get('risk_reward_ratio', 0):.2f}
- Description: {sig.get('description')}

NOTE: This is an INDEX instrument - volume data is not available/relevant.

CRITICAL: Your verdict MUST match the DIRECTION above:
- If DIRECTION is SHORT → use STRONG_SELL or CAUTIOUS_SELL (NOT BUY)
- If DIRECTION is LONG → use STRONG_BUY or CAUTIOUS_BUY (NOT SELL)

Evaluate based on Multi-Timeframe alignment and Market Structure.
"""


# Singleton access
_analyzer_instance = None


def get_analyzer():
    """Return singleton Vertex AI analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = VertexAnalyzer()
    return _analyzer_instance

```

## Telegram Module

### ./telegram_module/bot_handler.py

```python
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

```

## Cloud Functions

### ./retrain_function/main.py

```python
"""
Cloud Function for Weekly Model Retraining
Triggered by Cloud Scheduler
"""

import logging
from datetime import datetime
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from google.cloud import firestore

# Import from deployed package
from ml_module.feature_extractor import get_feature_names, get_categorical_features
from ml_module.model_storage import ModelStorage
from data_module.ml_data_collector import MLDataCollector
from config.settings import (
    ML_MODEL_BUCKET,
    ML_MIN_TRAINING_SAMPLES,
    GOOGLE_CLOUD_PROJECT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retrain_model(request):
    """
    Cloud Function entry point for weekly retraining.
    
    Args:
        request: Flask request object (unused)
        
    Returns:
        Response tuple (message, status_code)
    """
    logger.info("=" * 70)
    logger.info("🔄 Weekly Model Retraining Started")
    logger.info("=" * 70)
    
    try:
        # 1. Connect to Firestore
        db = firestore.Client(project=GOOGLE_CLOUD_PROJECT)
        collector = MLDataCollector(db)
        
        # 2. Fetch training data
        logger.info("Fetching training data from last 90 days...")
        training_records = collector.get_training_data(
            days=90,
            min_samples=ML_MIN_TRAINING_SAMPLES
        )
        
        if len(training_records) < ML_MIN_TRAINING_SAMPLES:
            msg = (
                f"Insufficient data: {len(training_records)} samples "
                f"(min: {ML_MIN_TRAINING_SAMPLES})"
            )
            logger.warning(f"⚠️ {msg}")
            return (msg, 200)  # Not an error, just not enough data yet
        
        # 3. Prepare data
        X, y = prepare_data(training_records)
        
        # 4. Train model
        model, score = train_quick_model(X, y)
        
        # 5. Validate performance
        if score < 0.55:  # Must beat random (0.5)
            logger.warning(
                f"⚠️ Model underperforming (AUC: {score:.4f}), "
                "keeping previous model"
            )
            return (f"Model performance too low: {score:.4f}", 200)
        
        # 6. Save to GCS
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            model.save_model(tmp.name)
            tmp_path = tmp.name
        
        storage = ModelStorage(bucket_name=ML_MODEL_BUCKET)
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if storage.upload_model(tmp_path, version=version):
            logger.info(f"✅ Model retrained and uploaded | AUC: {score:.4f}")
            return (f"Success: Model v{version} | AUC: {score:.4f}", 200)
        else:
            logger.error("❌ Failed to upload model to GCS")
            return ("Failed to upload model", 500)
    
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}", exc_info=True)
        return (f"Error: {str(e)}", 500)


def prepare_data(training_records: list) -> tuple:
    """Prepare X, y from Firestore records."""
    features_list = [r["features"] for r in training_records]
    labels = [r["label"] for r in training_records]
    
    X = pd.DataFrame(features_list)
    y = pd.Series(labels)
    
    logger.info(f"Dataset: {X.shape} | Win rate: {y.mean():.2%}")
    return X, y


def train_quick_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Quick training with minimal CV for Cloud Function timeout.
    
    Returns:
        (model, validation_score)
    """
    categorical_features = get_categorical_features()
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbose": -1
    }
    
    # Single train/val split (last 20% for validation)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    score = model.best_score["valid_0"]["auc"]
    logger.info(f"Validation AUC: {score:.4f}")
    
    return model, score

```

## Audit System

### ./audit_system.py

```python

import sys
import os
import logging
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import NiftyTradingAgent
from config.settings import INSTRUMENTS

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SystemAudit")

def create_mock_ohlcv(rows=500, trend="UP"):
    """Generate mock OHLCV data."""
    data = []
    price = 25000.0
    for i in range(rows):
        if trend == "UP":
            price += 10 if i % 2 == 0 else -5
        else:
            price -= 10 if i % 2 == 0 else 5
            
        high = price + 20
        low = price - 20
        close = price + 5
        volume = 100000 + (i * 100)
        
        data.append({
            "timestamp": datetime.now(),
            "open": price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df

def audit_system():
    logger.info("🚀 Starting System Health Audit...")
    
    try:
        # 1. Initialize Agent
        logger.info("Step 1: Initializing Agent...")
        agent = NiftyTradingAgent()
        logger.info("✅ Agent Initialized")

        # 2. Mock Data Fetcher
        logger.info("Step 2: Mocking Data Sources...")
        agent.fetcher.fetch_nse_data = MagicMock(return_value={"price": 25500, "lastPrice": 25500})
        agent.fetcher.fetch_historical_data = MagicMock(side_effect=[
            create_mock_ohlcv(500, "UP"), # 5m
            create_mock_ohlcv(500, "UP"), # 15m
            create_mock_ohlcv(50, "UP")   # Daily
        ])
        agent.fetcher.preprocess_ohlcv = MagicMock(side_effect=lambda x: x) # Passthrough

        # 3. Mock Option Fetcher
        logger.info("Step 3: Mocking Option Chain...")
        agent.option_fetcher.fetch_option_chain = MagicMock(return_value={
            "records": {
                "expiryDates": ["12-Dec-2024"],
                "data": []
            },
            "filtered": {
                "data": []
            }
        })
        
        # 4. Mock AI & Telegram (Don't want real calls)
        agent.groq_analyzer.analyze_signal = MagicMock(return_value={"verdict": "BULLISH", "confidence": 85, "reasoning": "Audit Test"})
        agent.telegram_bot.send_message = MagicMock(return_value=True)
        agent.telegram_bot.send_alert = MagicMock(return_value=True)

        # 5. Run Analysis
        logger.info("Step 4: Running Full Analysis Cycle...")
        results = agent.run_analysis(instruments=["NIFTY"])
        
        logger.info(f"📊 Audit Results: {results}")
        
        if results["errors"] > 0:
            logger.error(f"❌ Audit Failed with {results['errors']} errors!")
            sys.exit(1)
            
        if results["signals_generated"] == 0:
             logger.warning("⚠️ No signals generated (Check filters?)")
        else:
             logger.info(f"✅ Generated {results['signals_generated']} signals")

        logger.info("✅ System Audit Passed!")

    except Exception as e:
        logger.error(f"❌ CRITICAL AUDIT FAILURE: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    audit_system()

```

## Tests

### ./tests/__init__.py

```python
# Test Runner
# This file is intentionally empty to make tests directory a Python package

```

### ./tests/test_adaptive_thresholds.py

```python
"""
Unit Tests for Adaptive Thresholds Module
Tests VIX-based and ATR-based RSI threshold calculations
"""

import unittest
from analysis_module.adaptive_thresholds import AdaptiveThresholds
import pandas as pd
import numpy as np


class TestAdaptiveThresholds(unittest.TestCase):
    """Test suite for adaptive threshold calculations."""
    
    def test_vix_low_volatility(self):
        """Test RSI thresholds with low VIX (< 12)."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(vix=11.0)
        
        self.assertEqual(rsi_long, 55, "Low VIX should give tighter long threshold")
        self.assertEqual(rsi_short, 45, "Low VIX should give tighter short threshold")
    
    def test_vix_normal_volatility(self):
        """Test RSI thresholds with normal VIX (12-18)."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(vix=15.0)
        
        self.assertEqual(rsi_long, 60, "Normal VIX should give default long threshold")
        self.assertEqual(rsi_short, 40, "Normal VIX should give default short threshold")
    
    def test_vix_high_volatility(self):
        """Test RSI thresholds with high VIX (> 18)."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(vix=22.0)
        
        self.assertEqual(rsi_long, 65, "High VIX should give stricter long threshold")
        self.assertEqual(rsi_short, 35, "High VIX should give stricter short threshold")
    
    def test_atr_low_percentile(self):
        """Test RSI thresholds with low ATR percentile (< 30%)."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(
            vix=None, 
            atr_percentile=25.0
        )
        
        self.assertEqual(rsi_long, 55, "Low ATR percentile should give tighter thresholds")
        self.assertEqual(rsi_short, 45)
    
    def test_atr_normal_percentile(self):
        """Test RSI thresholds with normal ATR percentile (30-70%)."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(
            vix=None,
            atr_percentile=50.0
        )
        
        self.assertEqual(rsi_long, 60, "Normal ATR percentile should give default thresholds")
        self.assertEqual(rsi_short, 40)
    
    def test_atr_high_percentile(self):
        """Test RSI thresholds with high ATR percentile (> 70%)."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(
            vix=None,
            atr_percentile=80.0
        )
        
        self.assertEqual(rsi_long, 65, "High ATR percentile should give stricter thresholds")
        self.assertEqual(rsi_short, 35)
    
    def test_vix_priority_over_atr(self):
        """Test that VIX takes priority when both available."""
        # VIX says high volatility, ATR says low
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(
            vix=20.0,  # High VIX
            atr_percentile=25.0  # Low ATR
        )
        
        # Should use VIX thresholds (65/35), not ATR (55/45)
        self.assertEqual(rsi_long, 65, "VIX should take priority over ATR")
        self.assertEqual(rsi_short, 35)
    
    def test_atr_percentile_calculation(self):
        """Test ATR percentile calculation."""
        # Create sample dataframe with ATR values
        df = pd.DataFrame({
            'atr': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190] + [200] * 50
        })
        
        current_atr = 150
        percentile = AdaptiveThresholds.calculate_atr_percentile(df, current_atr, lookback=60)
        
        # 150 is higher than first 5 values (100-140), so percentile should be ~8.3%
        self.assertLess(percentile, 20, "Percentile calculation should work correctly")
        self.assertGreater(percentile, 5)
    
    def test_atr_threshold_calculation(self):
        """Test ATR threshold as percentile."""
        # Create sample dataframe
        df = pd.DataFrame({
            'atr': [50 + i for i in range(100)]
        })
        
        atr_threshold = AdaptiveThresholds.get_atr_threshold(df, atr_period=14)
        
        # 60th percentile of 50-149 should be around 109-120
        self.assertGreater(atr_threshold, 105, "ATR threshold should be meaningful")
        self.assertLess(atr_threshold, 125)
    
    def test_market_volatile_detection(self):
        """Test volatile market detection."""
        # High VIX
        self.assertTrue(
            AdaptiveThresholds.is_market_volatile(vix=20.0),
            "VIX > 18 should be detected as volatile"
        )
        
        # High ATR percentile
        self.assertTrue(
            AdaptiveThresholds.is_market_volatile(atr_percentile=75.0),
            "ATR > 70% should be detected as volatile"
        )
        
        # Normal VIX
        self.assertFalse(
            AdaptiveThresholds.is_market_volatile(vix=15.0),
            "VIX 15 should not be volatile"
        )
    
    def test_market_choppy_detection(self):
        """Test choppy market detection."""
        # Low VIX
        self.assertTrue(
            AdaptiveThresholds.is_market_choppy(vix=11.0),
            "VIX < 12 should be detected as choppy"
        )
        
        # Low ATR percentile
        self.assertTrue(
            AdaptiveThresholds.is_market_choppy(atr_percentile=25.0),
            "ATR < 30% should be detected as choppy"
        )
        
        # Normal VIX
        self.assertFalse(
            AdaptiveThresholds.is_market_choppy(vix=15.0),
            "VIX 15 should not be choppy"
        )
    
    def test_boundary_conditions(self):
        """Test boundary conditions for VIX and ATR."""
        # VIX exactly 12
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(vix=12.0)
        self.assertEqual(rsi_long, 60, "VIX 12 should use normal thresholds")
        
        # VIX exactly 18
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(vix=18.0)
        self.assertEqual(rsi_long, 60, "VIX 18 should use normal thresholds")
        
        # ATR exactly 30%
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(atr_percentile=30.0)
        self.assertEqual(rsi_long, 60, "ATR 30% should use normal thresholds")
        
        # ATR exactly 70%
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(atr_percentile=70.0)
        self.assertEqual(rsi_long, 60, "ATR 70% should use normal thresholds")
    
    def test_none_inputs(self):
        """Test behavior when both VIX and ATR are None."""
        rsi_long, rsi_short = AdaptiveThresholds.get_rsi_thresholds(vix=None, atr_percentile=None)
        
        # Should return default thresholds
        self.assertEqual(rsi_long, 60, "Should return default when both None")
        self.assertEqual(rsi_short, 40)


if __name__ == '__main__':
    unittest.main()

```

### ./tests/test_option_chain.py

```python

import unittest
import logging
from unittest.mock import MagicMock, patch
from analysis_module.option_chain_analyzer import OptionChainAnalyzer
from app.agent import NiftyTradingAgent

# Configure logging to show info during tests
logging.basicConfig(level=logging.INFO)

class TestOptionChainAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = OptionChainAnalyzer()
        # Mock Option Chain Data (Simplified structure)
        self.mock_data = {
            "records": {
                "data": [
                    {
                        "strikePrice": 26000,
                        "CE": {"openInterest": 100000, "changeinOpenInterest": 5000, "impliedVolatility": 12.0},
                        "PE": {"openInterest": 50000, "changeinOpenInterest": -2000, "impliedVolatility": 14.0}
                    },
                    {
                        "strikePrice": 26100, # ATM
                        "CE": {"openInterest": 80000, "changeinOpenInterest": 10000, "impliedVolatility": 11.5},
                        "PE": {"openInterest": 90000, "changeinOpenInterest": 15000, "impliedVolatility": 13.5}
                    },
                    {
                        "strikePrice": 26200,
                        "CE": {"openInterest": 60000, "changeinOpenInterest": -1000, "impliedVolatility": 11.0},
                        "PE": {"openInterest": 120000, "changeinOpenInterest": 20000, "impliedVolatility": 12.0}
                    }
                ]
            }
        }

    def test_pcr_calculation(self):
        """Test Put-Call Ratio Calculation"""
        # Total CE OI = 100k + 80k + 60k = 240k
        # Total PE OI = 50k + 90k + 120k = 260k
        # PCR = 260/240 = 1.0833
        pcr = self.analyzer.calculate_pcr(self.mock_data)
        self.assertAlmostEqual(pcr, 1.0833, places=2)
        print(f"\n✅ PCR Test Passed: Calculated {pcr}")

    def test_max_pain(self):
        """Test Max Pain Calculation"""
        # Max Pain logic is complex, verification of result:
        # At 26000: PE writers lose on nothing (ITM Puts only above). Calls below are ITM.
        # This is a functional test to ensure it returns a strike from the list.
        mp = self.analyzer.calculate_max_pain(self.mock_data)
        self.assertIn(mp, [26000, 26100, 26200])
        print(f"✅ Max Pain Test Passed: Calculated {mp}")

    def test_iv_calculation(self):
        """Test ATM IV Calculation"""
        spot = 26105 # Close to 26100
        iv = self.analyzer.calculate_atm_iv(self.mock_data, spot)
        # Expected: Avg of 11.5 and 13.5 = 12.5
        self.assertEqual(iv, 12.5)
        print(f"✅ IV Test Passed: Calculated {iv}")

    def test_oi_sentiment(self):
        """Test OI Change Sentiment"""
        spot = 26100
        # Call Chg: 5k + 10k - 1k = 14k
        # Put Chg: -2k + 15k + 20k = 33k
        # Put Chg (33k) > Call Chg (14k) * 1.5 -> BULLISH
        metrics = self.analyzer.analyze_oi_change(self.mock_data, spot)
        self.assertEqual(metrics["sentiment"], "BULLISH")
        print(f"✅ OI Sentiment Test Passed: {metrics['sentiment']}")


from datetime import datetime
from analysis_module.technical import Signal, SignalType

class TestSignalIntegration(unittest.TestCase):
    def setUp(self):
        self.agent = NiftyTradingAgent()
        self.agent.option_fetcher = MagicMock()
        self.agent.option_analyzer = OptionChainAnalyzer() # Use real logic

    def test_conflict_resolution_bullish_pcr(self):
        """Test that Bullish PCR resolves conflict in favor of LONG"""
        # Mock PCR > 1.2 (Bullish) via mock Option Data
        mock_oc_data = {
            "records": {"data": [
                 {"strikePrice": 26000, "CE": {"openInterest": 100}, "PE": {"openInterest": 200}}, # PCR=2.0
                 {"strikePrice": 26100, "CE": {"openInterest": 100}, "PE": {"openInterest": 200}}
            ]}
        }
        self.agent.option_fetcher.fetch_option_chain.return_value = mock_oc_data
        
        # Prepare conflicting objects
        # 1. Bullish Signal (e.g., Retest)
        bullish_sig = Signal(
            signal_type=SignalType.SUPPORT_BOUNCE,
            instrument="NIFTY",
            timeframe="5m",
            price_level=26000.0,
            entry_price=26010.0,
            stop_loss=25980.0,
            take_profit=26100.0,
            confidence=60.0,
            timestamp=datetime.now(),
            description="Bullish Bounce",
            risk_reward_ratio=2.0,
            volume_confirmed=True,
            momentum_confirmed=True
        )
        
        # 2. Bearish Signal (e.g., Engulfing) - Higher confidence to test override
        bearish_sig = Signal(
            signal_type=SignalType.BEARISH_ENGULFING,
            instrument="NIFTY",
            timeframe="5m",
            price_level=26050.0,
            entry_price=26040.0,
            stop_loss=26060.0,
            take_profit=25950.0,
            confidence=80.0, # Higher confidence
            timestamp=datetime.now(),
            description="Bearish Engulfing",
            risk_reward_ratio=2.0,
            volume_confirmed=True,
            momentum_confirmed=True
        )
        
        # Inject into analysis dict
        analysis_input = {
            "retest_signal": bullish_sig,
            "engulfing_signal": bearish_sig
        }
        
        nse_data_input = {"lastPrice": 26020, "price": 26020}
        
        # We need to ensure volume_check doesn't block it.
        # But looking at main.py, _generate_signals calls self.technical_analyzer only for breakout_signal logic?
        # Re-reading main.py:
        # lines 356 (breakout) -> checks volume_confirmed
        # lines 383 (retest) -> no extra checks inside _generate_signals logic itself usually?
        # Let's verify specific blocks in main.py logic for retest/engulfing.
        # It seems simple appending if confidence >= MIN.
        
        # Run
        signals = self.agent._generate_signals("NIFTY", analysis_input, nse_data_input)
        
        # Verify
        # Should have 1 signal
        self.assertEqual(len(signals), 1)
        # Should be the Bullish one despite lower confidence, because PCR=2.0 (Bullish)
        self.assertIn("SUPPORT_BOUNCE", signals[0]["signal_type"])
        print(f"\n✅ Conflict Resolution Passed: Kept {signals[0]['signal_type']} due to PCR")

if __name__ == '__main__':
    unittest.main()

```

### ./tests/test_risk_management.py

```python

import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from datetime import datetime

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_module.trade_tracker import TradeTracker

class TestRiskManagement(unittest.TestCase):
    
    @patch("data_module.trade_tracker.firestore.Client")
    def test_atr_trailing_stop_update(self, MockFirestore):
        # Setup Mock DB
        mock_db = MockFirestore.return_value
        mock_collection = mock_db.collection.return_value
        
        # Create Tracker
        tracker = TradeTracker()
        tracker.db = mock_db # Ensure it uses our mock
        
        # 1. Simulate an Open Trade (LONG)
        # Entry: 100, SL: 95, ATR: 2.0
        # Initial Trail Gap: 2.0 * 1.5 = 3.0
        
        trade_id = "test_trade_1"
        trade_data = {
            "trade_id": trade_id,
            "instrument": "NIFTY",
            "signal_type": "BULLISH_BREAKOUT",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "atr": 2.0,
            "status": "OPEN",
            "timestamp": datetime.now()
        }
        
        # Mock Query Result
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = trade_data
        mock_doc.reference = MagicMock() # The document reference for updating
        
        # Mock the stream() method to return our trade
        mock_collection.where.return_value.stream.return_value = [mock_doc]
        
        # 2. Case A: Price moves up to 105
        # New Potential SL = Price (105) - (ATR(2)*1.5) = 105 - 3 = 102
        # 102 > 95 (Old SL) -> SHOULD UPDATE
        
        current_prices = {"NIFTY": 105.0}
        tracker.check_open_trades(current_prices)
        
        # Verify Update Call
        mock_doc.reference.update.assert_called_with({"stop_loss": 102.0})
        print("\n✅ Case A: Trailing SL updated correctly (95.0 -> 102.0)")
        
        # Reset Mock
        mock_doc.reference.update.reset_mock()
        
        # 3. Case B: Price drops to 98 (Pullback)
        # New Potential SL = 98 - 3 = 95
        # 95 <= 95 (Old SL is effectively 95 or 102 depending on persistence, logic relies on db value)
        # Let's say DB still has 95 for this test because we didn't actually write to DB.
        # But if we assume we are testing logic: 
        #   Potential SL (95) is NOT > Old SL (95). So NO Update.
        
        current_prices = {"NIFTY": 98.0}
        tracker.check_open_trades(current_prices)
        
        mock_doc.reference.update.assert_not_called()
        print("✅ Case B: Trailing SL did NOT loosen on pullback")

    @patch("data_module.trade_tracker.firestore.Client")
    def test_short_trailing_update(self, MockFirestore):
        # Setup Mock DB
        mock_db = MockFirestore.return_value
        tracker = TradeTracker()
        tracker.db = mock_db
        
        # SHORT Trade
        # Entry: 100, SL: 105, ATR: 2.0
        # Trail Gap: 3.0
        trade_data = {
            "trade_id": "test_short",
            "instrument": "NIFTY",
            "signal_type": "BEARISH_BREAKOUT",
            "entry_price": 100.0,
            "stop_loss": 105.0,
            "take_profit": 90.0,
            "atr": 2.0,
            "status": "OPEN",
            "timestamp": datetime.now()
        }
        
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = trade_data
        mock_db.collection.return_value.where.return_value.stream.return_value = [mock_doc]
        
        # Price moves down (Profit) to 95
        # New Potential SL = 95 + 3 = 98
        # 98 < 105 (Old SL) -> SHOULD UPDATE
        
        current_prices = {"NIFTY": 95.0}
        tracker.check_open_trades(current_prices)
        
        mock_doc.reference.update.assert_called_with({"stop_loss": 98.0})
        print("\n✅ SHORT: Trailing SL tightened correctly (105.0 -> 98.0)")

if __name__ == '__main__':
    unittest.main()

```

### ./tests/test_scoring.py

```python

import unittest
import logging
from unittest.mock import MagicMock
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.agent import NiftyTradingAgent
from analysis_module.technical import Signal, SignalType
from analysis_module.option_chain_analyzer import OptionChainAnalyzer

# Logging
logging.basicConfig(level=logging.INFO)

class TestWeightedScoring(unittest.TestCase):
    def setUp(self):
        self.agent = NiftyTradingAgent()
        self.agent.option_fetcher = MagicMock()
        self.agent.option_analyzer = OptionChainAnalyzer()
        self.agent.fetcher = MagicMock()
        self.agent.fetcher.get_historical_data.return_value = None # For choppy check (mock it out)
        
        # Patch TechnicalAnalyzer inside main.py if needed, 
        # but _generate_signals creates a new instance.
        # We can mock `_is_choppy_session` on the class or just ensure it returns False.
        # Ideally, we mock TechnicalAnalyzer entirely.
        
    @unittest.mock.patch('analysis_module.technical.TechnicalAnalyzer')
    def test_scoring_logic_bullish(self, MockAnalyzer):
        """Test Scoring System for a Bullish Signal"""
        # 1. Setup Mock Analyzer to avoid Choppy Session
        mock_instance = MockAnalyzer.return_value
        mock_instance._is_choppy_session.return_value = (False, "Trend")
        
        # 2. Setup Option Chain Data (Bullish PCR)
        # PCR = 200/100 = 2.0 (Deep Bullish)
        mock_oc_data = {
            "records": {"data": [
                 {"strikePrice": 26100, "CE": {"openInterest": 100}, "PE": {"openInterest": 200}}
            ]}
        }
        self.agent.option_fetcher.fetch_option_chain.return_value = mock_oc_data
        
        # 3. Create a Signal (Weak Technicals but Strong Options)
        # Conf 65 (+10 pts)
        # Volume Confirmed (+10 pts)
        # Base = 50
        # Total from Technicals = 50 + 10 + 10 = 70
        # Options: PCR Bullish on Bullish Signal -> +10
        # Total Score = 80
        
        sig = Signal(
            signal_type=SignalType.BULLISH_BREAKOUT,
            instrument="NIFTY",
            timeframe="5m",
            price_level=26100.0,
            entry_price=26110.0,
            stop_loss=26080.0,
            take_profit=26200.0,
            confidence=65.0,
            timestamp=datetime.now(),
            description="Breakout",
            risk_reward_ratio=3.0,
            volume_confirmed=True,
            momentum_confirmed=True
        )
        
        analysis_input = {"breakout_signal": sig}
        nse_data_input = {"lastPrice": 26105, "price": 26105}
        
        # 4. Generate Signals
        signals = self.agent._generate_signals("NIFTY", analysis_input, nse_data_input)
        
        # 5. Verify Score
        self.assertTrue(len(signals) > 0, "Signal should be accepted")
        result_sig = signals[0]
        score = result_sig.get("score", 0)
        reasons = result_sig.get("score_reasons", [])
        
        print(f"\n✅ Result Score: {score}")
        print(f"📝 Reasons: {reasons}")
        
        # Assertions
        self.assertGreaterEqual(score, 70)
        self.assertIn("PCR Bullish", str(reasons))
        self.assertIn("Volume High (+10)", reasons)

    @unittest.mock.patch('analysis_module.technical.TechnicalAnalyzer')
    def test_scoring_rejection(self, MockAnalyzer):
        """Test Rejection of Weak Signal"""
        # 1. Setup Mock
        mock_instance = MockAnalyzer.return_value
        mock_instance._is_choppy_session.return_value = (False, "Trend")
        
        # 2. Setup Bearish Options for a Bullish Signal
        # PCR = 0.5 (Bearish)
        mock_oc_data = {
            "records": {"data": [
                 {"strikePrice": 26100, "CE": {"openInterest": 200}, "PE": {"openInterest": 100}}
            ]}
        }
        self.agent.option_fetcher.fetch_option_chain.return_value = mock_oc_data
        
        # 3. Weak Bullish Signal
        # Base = 50
        # Conf 55 (<65) -> +0 pts
        # Volume False -> +0 pts
        # Options: PCR Bearish (0.5) on Bullish Signal -> -10 pts
        # Total Score = 40
        # Threshold 60 -> REJECT
        
        sig = Signal(
            signal_type=SignalType.BULLISH_BREAKOUT,
            instrument="NIFTY",
            timeframe="5m",
            price_level=26100.0,
            entry_price=26110.0,
            stop_loss=26080.0,
            take_profit=26200.0,
            confidence=55.0,
            timestamp=datetime.now(),
            description="Weak Breakout",
            risk_reward_ratio=3.0,
            volume_confirmed=False,
            momentum_confirmed=True
        )
        
        analysis_input = {"breakout_signal": sig}
        nse_data_input = {"lastPrice": 26105, "price": 26105}
        
        # 4. Generate
        signals = self.agent._generate_signals("NIFTY", analysis_input, nse_data_input)
        
        # 5. Verify Rejection
        self.assertEqual(len(signals), 0, "Weak signal should be rejected by scoring")
        print("\n✅ Weak Signal Rejected as expected.")

    @unittest.mock.patch('analysis_module.technical.TechnicalAnalyzer')
    def test_mtf_alignment_boost(self, MockAnalyzer):
        """Test MTF Trend Alignment Boost (+15)"""
        # 1. Setup Mock (Aligned Trend)
        mock_instance = MockAnalyzer.return_value
        mock_instance._is_choppy_session.return_value = (False, "Trend")
        
        # 2. Options (Neutral)
        mock_oc_data = {
            "records": {"data": [
                 {"strikePrice": 26100, "CE": {"openInterest": 100}, "PE": {"openInterest": 100}}
            ]}
        }
        self.agent.option_fetcher.fetch_option_chain.return_value = mock_oc_data
        
        # 3. Moderate Signal
        # Base = 50
        # Conf 70 (+10)
        # Volume False
        # MTF Trend UP (Bullish) -> +15
        # Total = 75
        
        sig = Signal(
            signal_type=SignalType.BULLISH_BREAKOUT,
            instrument="NIFTY",
            timeframe="5m",
            price_level=26100.0,
            entry_price=26110.0,
            stop_loss=26080.0,
            take_profit=26200.0,
            confidence=70.0,
            timestamp=datetime.now(),
            description="Breakout",
            risk_reward_ratio=3.0,
            volume_confirmed=False,
            momentum_confirmed=True
        )
        
        # Inject Context logic via main.py modification assumption
        # Note: We need to verify that main.py logic picks this up.
        # Since we can't easily inject into the internal variable of _generate_signals without
        # using integration test style or modifying how we pass analysis,
        # we rely on the fact that _generate_signals receives 'analysis' dict.
        
        analysis_input = {
            "breakout_signal": sig,
            "higher_tf_context": {"trend_direction": "UP"}
        }
        nse_data_input = {"lastPrice": 26105, "price": 26105}
        
        signals = self.agent._generate_signals("NIFTY", analysis_input, nse_data_input)
        
        self.assertTrue(len(signals) > 0)
        self.assertGreaterEqual(signals[0]["score"], 75)
        self.assertIn("Trend Aligned", str(signals[0]["score_reasons"]))
        print(f"\n✅ MTF Alignment Boost Verified: Score {signals[0]['score']}")

    @unittest.mock.patch('analysis_module.technical.TechnicalAnalyzer')
    def test_mtf_conflict_penalty(self, MockAnalyzer):
        """Test MTF Trend Conflict Penalty (-15)"""
        # 1. Setup Mock
        mock_instance = MockAnalyzer.return_value
        mock_instance._is_choppy_session.return_value = (False, "Trend")
        
        # 2. Options (Neutral)
        # 3. Good Signal but Counter Trend
        # Base = 50
        # Conf 70 (+10)
        # Volume True (+10)
        # MTF Trend DOWN (Bearish vs Bullish Sig) -> -15
        # Total = 70 - 15 = 55 (REJECT < 60)
        
        mock_oc_data = {"records": {"data": [{"strikePrice": 26100, "CE": {"openInterest": 100}, "PE": {"openInterest": 100}}]}}
        self.agent.option_fetcher.fetch_option_chain.return_value = mock_oc_data
        
        sig = Signal(
            signal_type=SignalType.BULLISH_BREAKOUT,
            instrument="NIFTY",
            timeframe="5m",
            price_level=26100.0,
            entry_price=26110.0,
            stop_loss=26080.0,
            take_profit=26200.0,
            confidence=70.0,
            timestamp=datetime.now(),
            description="Counter Trend Breakout",
            risk_reward_ratio=3.0,
            volume_confirmed=True,
            momentum_confirmed=True
        )
        
        analysis_input = {
            "breakout_signal": sig,
            "higher_tf_context": {"trend_direction": "DOWN"}
        }
        nse_data_input = {"lastPrice": 26105, "price": 26105}
        
        signals = self.agent._generate_signals("NIFTY", analysis_input, nse_data_input)
        
        self.assertEqual(len(signals), 0, "Counter-trend signal should be rejected (Score ~55)")
        print("\n✅ MTF Conflict Penalty Verified: Signal Rejected")

if __name__ == '__main__':
    unittest.main()

```

