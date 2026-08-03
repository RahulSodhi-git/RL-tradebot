"""Backtest engine — supports long, short, leverage, and Binance fees (NET profit)."""
import numpy as np
import pandas as pd


def _trade_stats(close, pos):
    """Win rate over completed long AND short trades."""
    trades, entry, d = [], None, 0
    for i in range(len(pos)):
        if pos[i] != 0 and entry is None:                     # open
            entry, d = close[i], pos[i]
        elif entry is not None and pos[i] != d:               # close (flip or flat)
            trades.append(d * (close[i] - entry) / entry)
            entry, d = (close[i], pos[i]) if pos[i] != 0 else (None, 0)
    if entry is not None:                                     # close open trade at last price
        trades.append(d * (close[-1] - entry) / entry)
    t = np.array(trades)
    n = len(t)
    return n, (float((t > 0).mean() * 100) if n else 0.0)


def run(df, positions, leverage=1, fee=0.0005):
    """fee = taker fee per side (default 0.05% = Binance futures market order).
    Charged on entry AND exit, scaled by leverage (notional). Returns NET metrics."""
    pos = pd.Series(np.asarray(positions, float), index=df.index).shift(1).fillna(0)  # act next bar
    ret = df["close"].pct_change().fillna(0)
    cost = pos.diff().abs() * fee * leverage                  # fee on every entry/exit
    raw = pos * ret * leverage
    liquidated = bool((raw <= -1).any())                      # leverage wiped the position

    net_equity = (1 + (raw - cost).clip(lower=-1)).cumprod()  # after fees
    gross_equity = (1 + raw.clip(lower=-1)).cumprod()         # before fees
    peak = net_equity.cummax()
    n_trades, win = _trade_stats(df["close"].values, pos.values)
    net = float((net_equity.iloc[-1] - 1) * 100)
    gross = float((gross_equity.iloc[-1] - 1) * 100)
    return {
        "equity": net_equity,
        "return_pct": net,                                    # NET (after fees)
        "gross_return_pct": gross,
        "fees_pct": round(gross - net, 2),                    # profit eaten by fees
        "win_rate": win,
        "trades": n_trades,
        "max_dd": float(((net_equity - peak) / peak).min() * 100),
        "liquidated": liquidated,
    }
