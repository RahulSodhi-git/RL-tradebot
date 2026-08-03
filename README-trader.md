# TradeBot — Half 2: Binance Execution (Spot + Futures)

The AI agent's "hands" — connects to Binance and places real orders through an MCP server.
Half 1 decides *what* to trade; Half 2 *executes* it.

⚠️ **Trades real/fake money. Start on TESTNET. Never commit your `.env`.**

## Files
| File | Job |
|------|-----|
| `.env` | Your secret keys (you create this from `.env.example`) |
| `risk_rules.json` | Hard safety limits (max size, leverage, allowed coins, fail-safes) |
| `trader/client.py` | Talks to Binance (spot + futures) |
| `trader/guards.py` | Blocks unsafe orders in code |
| `mcp_server.py` | The tools the AI agent calls |
| `signals.py` | Bot brain: auto-pick best strategy + auto TP/SL |
| `bot.py` | Hands-off auto-trading loop |
| `test_connection.py` | Checks your keys work |

## Setup
```powershell
cd C:\Users\rsodh\PycharmProjects\tradebot
.\.venv\Scripts\Activate.ps1
pip install -r requirements-trader.txt
copy .env.example .env       # then open .env and paste your keys
python test_connection.py    # should print your balance + "Connection OK"
```

## Testnet keys (fake money — start here)
- Futures testnet: https://testnet.binancefuture.com  → API Key
- Spot testnet: https://testnet.binance.vision → API Key
Put them in `.env`, keep `USE_TESTNET=true`.

## The AI agent's tools (in `mcp_server.py`)
- `get_price(symbol)` — read price
- `account_balance()` — read balances
- `list_positions()` — open futures positions + PnL
- `spot_trade(symbol, side, usdt_amount)` — spot BUY/SELL
- `open_futures(symbol, side, usdt_amount, leverage)` — long/short with leverage
- `close_futures(symbol)` — close a position

## Hands-off auto-trading bot (`bot.py`)
Run once — it trades on its own until you stop it (Ctrl+C):
```powershell
python bot.py BTCUSDT 50      # coin, USDT per trade
```
Every cycle (`check_interval_seconds`) it:
1. Pulls fresh Binance data (completed candles) and backtests all 29 strategies
2. Picks the best one that is **reliable** (≥ `min_trades_for_strategy`) **and profitable**
3. On a Long/Short signal → opens the trade with your USDT + `default_leverage`
4. Auto-sizes take-profit / stop-loss from ATR volatility, **capped by fail-safes**
5. Places a **real stop-loss + take-profit on the exchange** (fires even if the bot is off)
6. Closes on take-profit, stop-loss, or a strategy flip — then repeats

If nothing is reliable + profitable, it **stays flat** (safe). State is saved in `bot_state.json`.

## Safety
- `USE_TESTNET=true` → fake money. `DRY_RUN=true` → orders only printed, not sent.
- `risk_rules.json` limits:
  - `max_usdt_per_trade`, `max_leverage`, `allowed_symbols`, `trading_enabled` (kill switch)
  - `min_trades_for_strategy` — reliability filter (ignore lucky flukes)
  - `default_leverage`, `check_interval_seconds`
  - `failsafe_max_stop_loss_pct`, `failsafe_max_take_profit_pct` — auto TP/SL can only get *safer*, never riskier
- Keys live only in `.env` (gitignored). Withdrawals must stay OFF on the key.
