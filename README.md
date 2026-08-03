# TradeBot — Half 1: Strategy Analyzer

A simple, clean web app that backtests **29 trading strategies** (long/short + leverage) on
real Binance data and shows **which strategy gives the best profit chance (win rate %)**.

- ✅ Uses **public** market data only — **no API key, no trading, no risk**.
- ✅ Analysis only. (Half 2 = execution + auto-bot — see `README-trader.md`.)

## Setup (one time)

```powershell
cd C:\Users\rsodh\PycharmProjects\tradebot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

A browser tab opens. Pick a coin + timeframe, press **Run Backtest**, and read the results.

## What each file does

| File | Job |
|------|-----|
| `app.py` | The web UI + results page (filter/sort, ranked table, charts) |
| `data.py` | Fetches candles from Binance public API (spot + futures) |
| `indicators.py` | Technical indicators (Supertrend, ADX, ATR, RSI, MACD, …) |
| `strategies.py` | The 29 strategies (Basic / Medium / Advanced) |
| `backtest.py` | Runs each strategy on history — return, win rate, trades, drawdown, leverage |

## Strategies included (29)

- **Basic (9):** SMA Cross, EMA Cross, EMA Fast, Price vs 200 EMA, RSI, Stochastic, Williams %R, Momentum, Buy & Hold
- **Medium (11):** MACD, Bollinger (Mean-Rev + Breakout), CCI, Parabolic SAR, Keltner, Donchian+ADX, VWAP, Heikin-Ashi, TRIX, Awesome Oscillator
- **Advanced (9):** Supertrend, Supertrend Fusion, Regime-Adaptive, Supertrend+RSI+Volume, Ichimoku, MACD+RSI, ATR Breakout, Dual Momentum, Z-Score

Every strategy can go **long and short**, and the backtest supports **leverage**.

> ⚠️ Backtests show past performance only. Not financial advice.

## The two halves
- **Half 1 — Analysis** (this file): backtest + rank strategies. No keys, no risk.
- **Half 2 — Execution + Auto-bot** (`README-trader.md`): connect to Binance, place orders via an MCP server, and run a hands-off auto-trading bot on testnet.
