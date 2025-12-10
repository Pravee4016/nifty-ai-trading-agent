# Nifty AI Trading Agent: Simplified Overview

## 🤖 What is it?
Think of this agent as a **tireless, automated trading assistant** that watches the stock market (specifically **NIFTY 50** and **BANK NIFTY**) for you all day long, from 9:15 AM to 3:30 PM.

It never gets tired, never acts on emotion, and only speaks up when it sees a high-probability trading opportunity.

## 🎯 What does it do?
Its main job is to **hunt for profitable setups** and send you an instant alert on **Telegram**.

It looks for specific "patterns" in the price chart, such as:
*   **Breakouts:** When price aggressively breaks through a ceiling (Resistance) or floor (Support).
*   **Reversals:** When price hits a wall and turns around (Pin Bars, Engulfing Candles).
*   **Trend Continuation:** Jumping on a moving train (Inside Bars).

## ⚙️ How does it work?

Imagine a team of experts working together inside a computer:

### 1. The Scout (Data Fetcher) 🕵️
Every **5 minutes**, this module wakes up and pulls the latest price data, volume, and Option Chain data from the NSE (National Stock Exchange) via the Fyers API.

### 2. The Analyst (Technical Analysis) 📉
This module crunches the numbers. It draws invisible lines on the chart (Support & Resistance), checks the trend direction, measures momentum (RSI), and looks for those specific patterns mentioned above.
*   *Example:* "Hey, Nifty just crossed above the day's high with strong volume!"

### 3. The Gatekeeper (Risk Manager) 🛡️
Before getting excited, this module checks the safety rules:
*   Is the market too "choppy" or sideways? (If so, ignore).
*   Did we already send too many alerts today?
*   Is the risk-to-reward ratio good enough? (i.e., is the potential profit worth the risk?)

### 4. The Wise Mentor (AI Brain) 🧠
If the setup passes the Analyst and the Gatekeeper, it is sent to an **Artificial Intelligence** (Groq metrics).
The AI acts like a senior trader. It reads the technical data and gives a second opinion:
*   "This looks good because the trend is up and volume is high."
*   "Be careful, there is a major resistance level just above."

### 5. The Messenger (Telegram Bot) 📨
Finally, if everyone agrees (Scout + Analyst + Gatekeeper + Mentor), the agent sends a **formatted message to your Telegram**.
This message tells you:
*   **What to buy** (e.g., NIFTY 50 Call Option).
*   **Entry Price** (Where to get in).
*   **Stop Loss** (Where to get out if wrong).
*   **Targets** (Where to book profit).
*   **AI Analysis** (Why this trade was chosen).
