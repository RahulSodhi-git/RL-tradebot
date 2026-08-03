"""The bot's brain: pick the best reliable strategy, read its live signal, and
auto-size take-profit / stop-loss from volatility (capped by fail-safe limits)."""
import numpy as np

from strategies import STRATEGIES
from backtest import run
from indicators import atr, ema


def higher_tf_trend(df):
    """+1 if the higher-timeframe trend is UP (EMA20 > EMA50), else -1.
    Used to only trade WITH the bigger trend (avoids counter-trend whipsaw)."""
    return 1 if ema(df["close"], 20).iloc[-1] > ema(df["close"], 50).iloc[-1] else -1


def pick_best_strategy(df, min_trades=20, max_return=100000):
    """Best reliable + profitable strategy, PLUS how many strategies confirm its signal.
    Skips 'miracle' overfit backtests (return above max_return).
    Returns a dict {name, fn, ret, trades, win, signal, confirmations} or None."""
    best = None
    sigs = {}
    for name, (fn, level) in STRATEGIES.items():
        if name == "Buy & Hold":
            continue
        pos = np.asarray(fn(df), float)
        sigs[name] = int(pos[-1])                     # each strategy's latest vote
        r = run(df, pos)
        if r["trades"] >= min_trades and 0 < r["return_pct"] <= max_return:
            if best is None or r["return_pct"] > best["ret"]:
                best = {"name": name, "fn": fn, "ret": r["return_pct"],
                        "trades": r["trades"], "win": r["win_rate"], "signal": int(pos[-1])}
    if best is None:
        return None
    best["confirmations"] = sum(1 for v in sigs.values() if v == best["signal"])
    return best


def latest_signal(fn, df):
    """+1 = go long, -1 = go short, 0 = stay flat (the strategy's newest reading)."""
    return int(np.asarray(fn(df), float)[-1])


def auto_leverage(df, positions, rules):
    """Pick the leverage (up to max_leverage) that MAXIMISES the backtested net return.
    Higher leverage amplifies profit, so this walks up while the backtest keeps improving and
    never liquidates — a leverage that would have wiped the position out is never chosen."""
    cap = rules.get("max_leverage", 5)
    best_lev = 1
    best_ret = run(df, positions, leverage=1)["return_pct"]
    for cand in (2, 3, 5):
        if cand > cap:
            break
        r = run(df, positions, leverage=cand)
        if not r["liquidated"] and r["return_pct"] > best_ret:   # keep the most profitable, safe leverage
            best_lev, best_ret = cand, r["return_pct"]
    return best_lev


def auto_tp_sl(df, side, entry_price, rules, leverage=1):
    """Stop-loss price = a wide price stop that SCALES DOWN with leverage, so the money at risk
    stays ~stop_loss_pct% of the margin at ANY leverage (20% price @1x, 4% @5x) — never a
    liquidation-level stop. The trailing stop does the profit-taking; this SL is the disaster floor.
    side: +1 long / -1 short."""
    atr_pct = float(atr(df, 14).iloc[-1]) / entry_price * 100
    sl_pct = rules.get("stop_loss_pct", 20.0) / max(1, leverage)      # scale by leverage = constant money risk
    tp_pct = min(3.0 * atr_pct, rules["failsafe_max_take_profit_pct"])  # (unused — trailing exits)
    sl_pct, tp_pct = max(sl_pct, 0.3), max(tp_pct, 0.6)                 # never zero
    if side > 0:  # long
        tp, sl = entry_price * (1 + tp_pct / 100), entry_price * (1 - sl_pct / 100)
    else:         # short
        tp, sl = entry_price * (1 - tp_pct / 100), entry_price * (1 + sl_pct / 100)
    return tp, sl, round(tp_pct, 2), round(sl_pct, 2)
