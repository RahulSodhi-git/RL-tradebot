"""Honest backtest report for the 7 / 1 / 2 slot design.

WHAT IS AND ISN'T BACKTESTABLE (read this):
  - NORMAL (7 slots): fully backtestable. Uses the SAME engine (backtest.run) the live bot
    uses to pick strategies, over real price history. Numbers below are real.
  - SPECIAL (2 slots): only a PROXY. The live entry also listens to "news / Binance Square /
    chatrooms" — there is NO historical dataset for social chatter, so that part cannot be
    replayed. This proxy fires ONLY on the volume-spike + momentum leg, then applies the exact
    money-management you asked for (10x, 2% hard stop, +10% target -> bank 50%, else +24h ->
    bank 75%). The runner is closed at breakeven / a max-hold, so its UPSIDE IS UNDERSTATED.
  - NEWS (1 slot): NOT backtestable here. news.py reads LIVE RSS; the project keeps no archive
    of timestamped past headlines, so there is nothing to replay. Forward/paper-test only.

Run:  python backtest_report.py
"""
import numpy as np

from data import get_ohlcv
from strategies import STRATEGIES
from backtest import run

UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
            "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT",
            "ATOMUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT",
            "INJUSDT", "TIAUSDT"]

FEE = 0.0005            # 0.05% taker per side (Binance futures market order)


# ----------------------------------------------------------------------------- NORMAL (real)
def backtest_normal(coins, tf="4h", bars=500, min_trades=20, max_return=300):
    """Per coin, pick the best reliable+profitable strategy over real history (same rule the
    live bot uses) and report its net (after-fee) backtest at 1x — how normal trades trade."""
    rows = []
    for c in coins:
        try:
            df = get_ohlcv(c, tf, bars, closed_only=True)
        except Exception:
            continue
        if df is None or len(df) < 100:
            continue
        best = None
        for name, (fn, _lvl) in STRATEGIES.items():
            if name == "Buy & Hold":
                continue
            try:
                res = run(df, np.asarray(fn(df), float), fee=FEE)
            except Exception:
                continue
            if res["trades"] >= min_trades and 0 < res["return_pct"] <= max_return:
                if best is None or res["return_pct"] > best["net"]:
                    best = {"coin": c, "strategy": name, "net": res["return_pct"],
                            "win": res["win_rate"], "trades": res["trades"],
                            "dd": res["max_dd"], "fees": res["fees_pct"]}
        if best:
            rows.append(best)
    return rows


# --------------------------------------------------------------- SPECIAL (volume-spike proxy)
def backtest_special(coins, tf="1h", bars=1000, lev=10, stop_pct=2.0, target_pct=10.0,
                     deadline_h=24, part_target=0.50, part_deadline=0.75, max_hold_h=72,
                     vol_spike=2.0, base_win=5, mom_lookback=3):
    """Mechanical proxy of the special money-management on volume-spike entries.
    PnL is expressed in % of the slot's margin (so a full 2% stop-out = -stop_pct*lev = -20%)."""
    fee_pct = 2 * FEE * lev * 100                       # entry + all exits ~= 2 sides, in % margin
    trades = []
    for c in coins:
        try:
            df = get_ohlcv(c, tf, bars, closed_only=True)
        except Exception:
            continue
        if df is None or len(df) < base_win + mom_lookback + 5:
            continue
        o, h, l, cl, v = (df["open"].values, df["high"].values, df["low"].values,
                          df["close"].values, df["volume"].values)
        n = len(df)
        i = base_win + mom_lookback
        while i < n - 1:
            base = v[i - base_win:i].mean()
            spike = v[i] / base if base > 0 else 0
            if spike < vol_spike:
                i += 1
                continue
            direction = 1 if cl[i] / cl[i - mom_lookback] - 1 >= 0 else -1
            entry = cl[i]
            stop = entry * (1 - stop_pct / 100) if direction > 0 else entry * (1 + stop_pct / 100)
            target = entry * (1 + target_pct / 100) if direction > 0 else entry * (1 - target_pct / 100)
            realized, rem, exit_stop, partialed = 0.0, 1.0, stop, False

            def hit_stop(j, s):
                return (l[j] <= s) if direction > 0 else (h[j] >= s)

            def hit_target(j):
                return (h[j] >= target) if direction > 0 else (l[j] <= target)

            def ret_at(px, frac):                        # % of margin from closing `frac` at `px`
                return frac * direction * (px / entry - 1) * lev * 100

            j = i + 1
            closed = False
            while j < n:
                age = j - i                              # hours (1h bars)
                if not partialed:
                    if hit_stop(j, stop):                # stopped before any profit -> full -20%
                        realized += ret_at(stop, 1.0); rem = 0.0; closed = True; break
                    if age <= deadline_h and hit_target(j):     # +target in time -> bank 50%
                        realized += ret_at(target, part_target)
                        rem -= part_target; partialed = True; exit_stop = entry   # runner -> breakeven
                    elif age >= deadline_h:              # deadline, target missed -> bank 75%
                        realized += ret_at(cl[j], part_deadline)
                        rem -= part_deadline; partialed = True; exit_stop = stop  # runner keeps 2% stop
                else:                                    # runner
                    if hit_stop(j, exit_stop):
                        realized += ret_at(exit_stop, rem); rem = 0.0; closed = True; break
                    if age >= max_hold_h:
                        realized += ret_at(cl[j], rem); rem = 0.0; closed = True; break
                j += 1
            if not closed and rem > 0:                   # ran out of data -> mark to last close
                realized += ret_at(cl[-1], rem)
                j = n - 1
            trades.append(realized - fee_pct)
            i = j + 1                                    # no overlapping trades in one slot
    return trades


def _p(label, xs):
    if not xs:
        print(f"  {label}: (no qualifying trades/coins)")
        return
    a = np.array(xs, float)
    wins = (a > 0).mean() * 100
    print(f"  {label}: n={len(a)} | avg {a.mean():+.1f}% | median {np.median(a):+.1f}% | "
          f"win {wins:.0f}% | best {a.max():+.1f}% | worst {a.min():+.1f}%")


if __name__ == "__main__":
    print("=" * 78)
    print("BACKTEST REPORT  —  7 normal / 1 news / 2 special   (each slot = 10% of capital)")
    print("=" * 78)

    print("\n[1] NORMAL (7 slots) — REAL backtest, best strategy per coin, 4h, ~500 bars, 1x, net of fees")
    normal = backtest_normal(UNIVERSE)
    normal.sort(key=lambda x: x["net"], reverse=True)
    for r in normal:
        print(f"    {r['coin']:<10} {r['strategy']:<26} net {r['net']:+7.1f}% | "
              f"win {r['win']:.0f}% | {r['trades']:>3} trades | maxDD {r['dd']:.0f}%")
    _p("NORMAL summary (per-coin net %)", [r["net"] for r in normal])
    if normal:
        prof = sum(1 for r in normal if r["net"] > 0)
        print(f"  coins with a profitable best-strategy: {prof}/{len(normal)} | "
              f"avg win-rate {np.mean([r['win'] for r in normal]):.0f}% | "
              f"avg maxDD {np.mean([r['dd'] for r in normal]):.0f}%")

    print("\n[2] SPECIAL (2 slots) — PROXY only (volume-spike leg; social/chatroom NOT replayable)")
    print("    10x, 2% hard stop, +10% target->bank 50%, else +24h->bank 75%, runner->breakeven/72h")
    print("    PnL = % of the slot's margin (a full stop-out = -20%). Runner upside is UNDERSTATED.")
    special = backtest_special(UNIVERSE)
    _p("SPECIAL summary (per-trade % on margin)", special)
    if special:
        a = np.array(special, float)
        print(f"  total of all special trades: {a.sum():+.0f}%  (sum of per-trade margin returns)")
        print(f"  full stop-outs (~-20%): {(a < -18).sum()} / {len(a)}")

    print("\n[3] NEWS (1 slot) — NOT BACKTESTED")
    print("    news.py reads LIVE RSS; there is no archive of past timestamped headlines to replay.")
    print("    Honest option: forward/paper-test on testnet, or buy a historical crypto-news dataset.")
    print("=" * 78)
