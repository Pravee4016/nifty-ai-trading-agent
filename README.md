# Nifty AI Trading Agent

Intraday Nifty / BankNifty breakout agent with:

- 5m entries, 15m trend confirmation.
- PDH/PDL + support / resistance clustering.
- Volume and RSI filters.
- False breakout detection.
- Groq AI summary of signals.
- Telegram alerts.

## Structure

- config/settings.py – config, thresholds, API keys (via .env).
- data_module/fetcher.py – real-time + historical OHLCV (Nifty, BankNifty).
- analysis_module/technical.py – PDH/PDL, S/R, breakouts, false breakouts, retests, inside bars.
- ai_module/groq_analyzer.py – Groq API client for signal evaluation.
- telegram_module/bot_handler.py – Telegram alerts and notifications.
- main.py – orchestrator (run once or as cloud function).
- requirements.txt – dependencies.
- .env.example – sample env vars.

## Setup

