"""Hands-off auto-trading bot (TESTNET).

You run it once:   python bot.py BTCUSDT 50
Then it loops forever:
  - backtest all strategies -> pick the best RELIABLE + PROFITABLE one
  - when that strategy signals Long/Short -> open the trade with your USDT
  - auto-set take-profit / stop-loss (capped by fail-safes) as REAL exchange orders
  - close on take-profit, stop-loss, or a strategy flip
Stop it anytime with Ctrl+C.
"""
import json
import os
import sys
import time

import numpy as np

from data import get_ohlcv
from indicators import adx
from strategies import STRATEGIES
from signals import pick_best_strategy, latest_signal, auto_tp_sl, auto_leverage, higher_tf_trend

_HIGHER_TF = {"1m": "15m", "3m": "1h", "5m": "1h", "15m": "4h", "30m": "4h", "1h": "4h",
              "2h": "1d", "4h": "1d", "6h": "1d", "8h": "1d", "12h": "1d", "1d": "1w", "3d": "1w", "1w": "1M"}
from trader.client import BinanceTrader
from trader import guards

STATE_FILE = "bot_state.json"
TIMEFRAME = "1h"
HISTORY = 500
LOSS_STREAK = {}   # coin -> consecutive losing trades (for the "stop after N losses" rule)
LAST_CLOSE = {}    # coin -> (epoch_closed, timeframe) for the re-entry cooldown
RECENT_OPENS = []  # timestamps of recent opens (spreads new trades out over time)


def _can_open_now(r):
    """Limit NEW trades per time window so the bot doesn't fire them all at once."""
    window = r.get("open_batch_window_min", 30) * 60
    now = time.time()
    RECENT_OPENS[:] = [ts for ts in RECENT_OPENS if now - ts < window]   # keep only recent
    return len(RECENT_OPENS) < r.get("max_trades_per_batch", 3)

_TF_MIN = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
           "4h": 240, "6h": 360, "8h": 480, "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080}


def _on_cooldown(symbol, r):
    """True if the coin was closed too recently (cooldown scaled to its timeframe)."""
    info = LAST_CLOSE.get(symbol)
    if not info:
        return False
    closed_at, tf = info
    cooldown_min = max(r.get("reentry_cooldown_min", 30),
                       _TF_MIN.get(tf, 240) * r.get("reentry_cooldown_factor", 1.0))
    return (time.time() - closed_at) < cooldown_min * 60


def rules():
    with open("risk_rules.json") as f:
        return json.load(f)


def load_state():
    return json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}


def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), indent=2)


def save_recommendations(picks):
    """Save the latest scan so the dashboard can show live suggestions (no button)."""
    try:
        keep = ("symbol", "timeframe", "strategy", "signal", "net", "confirmations",
                "leverage", "vol_spike", "volume_flag", "win", "trades", "trend")
        rows = [{k: p.get(k) for k in keep} for p in picks[:12]]
        json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "picks": rows},
                  open("recommendations.json", "w"))
    except Exception:
        pass


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:                     # Windows cp1252 console can't print some chars
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)


ACTIVITY_FILE = "activity.log"


def note(msg):
    """Plain-English diary of what the bot did, one simple sentence per line.
    Written to activity.log so a human can read exactly what happened and why."""
    try:
        with open(ACTIVITY_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
    except Exception:
        pass


def _coin(sym):
    """BTCUSDT -> BTC (nicer to read in the diary)."""
    return sym[:-4] if sym.endswith("USDT") else sym


def _fit_to_capital(t, want, r):
    """Size a trade at the requested 10%-of-wallet, but never ask for more MARGIN than is actually
    FREE right now — so when the slots are nearly full, a pending trade uses whatever capital is left
    instead of failing with -2019 'margin insufficient'. Returns the usable size, or None if there
    isn't even a minimum trade's worth of free margin. Keeps the 10% rule in profit AND loss (the
    caller already sized off the live wallet); this only caps it down to the free balance."""
    try:
        avail = t.futures_available()
    except Exception:
        return round(want, 2)                              # can't measure -> trust the requested size
    size = min(want, avail * r.get("capital_safety_frac", 0.95))   # leave a little for fees/slippage
    return round(size, 2) if size >= r.get("min_trade_usdt", 10) else None


def _record_close(pos, symbol, exit_price, reason):
    """Write a full row to trade_history.csv when a trade closes."""
    try:
        from history import record
        entry, side, lev = pos["entry"], pos["side"], pos.get("leverage", 1)
        move = (exit_price / entry - 1) * 100 * side
        pnl = round(move * lev, 2)
        LOSS_STREAK[symbol] = (LOSS_STREAK.get(symbol, 0) + 1) if pnl < 0 else 0   # win resets
        LAST_CLOSE[symbol] = (time.time(), pos.get("timeframe", "4h"))             # start re-entry cooldown
        why = {"trailing-stop": "the trailing stop locked in the move",
               "signal-flip": "the strategy changed its mind and now says the opposite",
               "closed on exchange": "the position disappeared from the exchange "
                                     "(stop-loss filled, or it was closed by hand)"}.get(reason, reason)
        outcome = "PROFIT" if pnl >= 0 else "LOSS"
        note(f"Closed {_coin(symbol)} ({'long' if side > 0 else 'short'}) at {exit_price:.6g} - "
             f"{outcome} of {pnl:+.2f}%. Reason: {why}.")
        frac_left = pos.get("remaining_frac", 1.0)        # after a target-partial only 50% is still open
        try:
            size_left = round(float(pos.get("size", 100)) * frac_left, 2)
        except (TypeError, ValueError):
            size_left = 100.0
        record({
            "open_time": pos.get("open_time", ""),
            "close_time": time.strftime("%Y-%m-%d %H:%M"),
            "coin": symbol, "side": "LONG" if side > 0 else "SHORT",
            "type": "special" if pos.get("special") else ("news" if pos.get("news") else "strategy"),
            "strategy": pos.get("strategy") or ("special" if pos.get("special") else "news"),
            "timeframe": pos.get("timeframe", ""),
            "leverage": lev, "confirmations": pos.get("confirmations", ""),
            "expected_return_%": pos.get("expected_return", ""),
            "entry": round(entry, 6), "exit": round(exit_price, 6),
            "price_move_%": round(move, 2), "pnl_%": pnl,
            "pnl_usdt": round(size_left * pnl / 100, 2),   # REAL $ on the margin STILL open (runner = half)
            "size_usdt": size_left,                        # margin still deployed at close
            "exit_reason": reason,
        })
    except Exception as e:
        log(f"history record failed ({symbol}): {e}")


def _trail_level(pos, price, r):
    """Update the peak and work out where the trailing stop currently sits.
    Shared by the fast 1s watcher AND the slow loop so they can never disagree.
    Returns (stop_price, armed, peak)."""
    side, entry = pos["side"], pos["entry"]
    peak = pos.get("peak", entry)
    peak = max(peak, price) if side > 0 else min(peak, price)   # best price reached so far
    pos["peak"] = peak
    peak_profit = (peak / entry - 1) * 100 * side
    stop = pos["sl"]                                            # below activation: hard SL only
    armed = peak_profit >= r.get("trail_activation_pct", 2.0)
    if armed:                                                   # trailing kicks in after +act% profit
        trail = (r.get("trail_pct_tight", 1.2) if peak_profit >= r.get("trail_tighten_at_pct", 5.0)
                 else r.get("trail_pct_normal", 2.0))           # tighten once deep in profit
        stop = (max(pos["sl"], peak * (1 - trail / 100)) if side > 0
                else min(pos["sl"], peak * (1 + trail / 100)))
        # BREAKEVEN CLAMP: once armed (>= +2% profit), never trail the stop into a loss —
        # the worst a trailing exit can do is breakeven (a fast market fill can still slip a hair).
        stop = max(stop, entry) if side > 0 else min(stop, entry)
    return stop, armed, peak


def watch_positions(t, r, state):
    """FAST loop (runs every second): live prices + trailing maths only.
    No downloads, no backtests. One API call covers every open trade.
    This is what closes a winner the moment it turns, instead of ~2.5 min later."""
    if not state:
        return
    try:
        prices = t.futures_prices_all()          # ONE request for all symbols
    except Exception:
        return                                   # network blip: the slow loop still covers us
    for sym in list(state.keys()):
        pos, price = state.get(sym), prices.get(sym)
        if not pos or not price:
            continue
        if pos.get("special"):                   # special: 1s-check the +target% window AND runner-trail
            try:
                s = pos["side"]; e = pos["entry"]
                pk = pos.get("peak", e)
                pos["peak"] = max(pk, price) if s > 0 else min(pk, price)   # track high-water mark
                if not _special_take_target(t, sym, pos, price, r, state):
                    _special_runner_trail(t, sym, pos, price, r, state)    # ride the runner on the wide ladder
            except Exception as e:
                log(f"fast special {sym}: {e}")
            continue
        try:
            old_peak = pos.get("peak", pos["entry"])
            stop, armed, peak = _trail_level(pos, price, r)
            side = pos["side"]
            # only the TRAILING exit here - the 20% hard stop is a real order on the exchange
            if armed and (price <= stop if side > 0 else price >= stop):
                _record_close(pos, sym, price, "trailing-stop")
                t.close_futures(sym)
                t.cancel_open_orders(sym)
                state.pop(sym, None)
                save_state(state)
                log(f"CLOSE {sym} @ {price:.6g}  (trailing-stop, 1s watch)")
            elif peak != old_peak:
                save_state(state)                # remember the new high water mark
        except Exception as e:
            log(f"fast watch {sym}: {e}")


def step(t, symbol, usdt, r, state, timeframe=TIMEFRAME):
    """One decision cycle: manage the open trade, or look for a new one."""
    if state.get(symbol, {}).get("special"):        # special trades run their OWN lifecycle
        _manage_special(t, symbol, r, state)        # (2% hard stop + timed partial TP), not this
        return
    min_trades = r.get("min_trades_for_strategy", 20)

    df = get_ohlcv(symbol, timeframe, HISTORY, closed_only=True)  # fresh data, completed candles only
    price = t.futures_price(symbol)
    pos = state.get(symbol)

    # If the exchange already closed us (TP/SL hit while we slept), clear state.
    live = {p["symbol"] for p in t.futures_positions()}
    if pos and symbol not in live:
        log(f"{symbol}: position already closed on exchange (TP/SL hit). Resetting.")
        _record_close(pos, symbol, price, "closed on exchange")
        t.cancel_open_orders(symbol)
        state.pop(symbol, None)
        save_state(state)
        pos = None

    if pos:  # ---- manage the open trade (TRAILING stop) ----
        side = pos["side"]
        if not pos.get("stop_placed", True):          # earlier stop-order failed -> keep retrying
            if "placed" in str(t.place_stops(symbol, side, None, pos["sl"])):
                pos["stop_placed"] = True
                save_state(state)
                log(f"{symbol}: re-placed the missing stop-loss on the exchange")
        strat = pos.get("strategy")
        if strat and strat in STRATEGIES:
            cur = latest_signal(STRATEGIES[strat][0], df)
            flipped = cur != side and cur != 0
        else:
            flipped = False               # news trade: no strategy -> exit only on trailing stop
        old_peak = pos.get("peak", pos["entry"])
        stop, armed, peak = _trail_level(pos, price, r)         # same maths as the fast watcher
        hit = price <= stop if side > 0 else price >= stop

        if peak != old_peak:
            save_state(state)
        reason = "trailing-stop" if hit else ("signal-flip" if flipped else None)
        if reason:
            _record_close(pos, symbol, price, reason)
            t.close_futures(symbol)
            t.cancel_open_orders(symbol)
            state.pop(symbol, None)
            save_state(state)
            log(f"CLOSE {symbol} @ {price:.4f}  ({reason})")
        else:
            tag = "LONG" if side > 0 else "SHORT"
            pl = (price / pos["entry"] - 1) * 100 * side
            log(f"HOLD {symbol} {tag} @ {price:.4f} | {'trailing' if armed else 'SL-only'} stop {stop:.4f} "
                f"| peak {peak:.4f} | P/L {pl:+.1f}% | {pos.get('strategy', 'news')}")

    else:  # ---- flat: look for a new trade ----
        if LOSS_STREAK.get(symbol, 0) >= r.get("max_losses_per_coin", 2):
            log(f"{symbol}: blocked - {LOSS_STREAK[symbol]} losses in a row, not trading it again")
            note(f"Skipped {_coin(symbol)} - it already lost {LOSS_STREAK[symbol]} times in a row.")
            return
        if _on_cooldown(symbol, r):
            log(f"{symbol}: re-entry cooldown active (recently closed) -> skip")
            return
        # CHOP FILTER: trend strength must be high enough, else strategies whipsaw (signal flips)
        min_adx = r.get("min_adx", 0)
        if min_adx:
            try:
                cur_adx = float(adx(df).iloc[-1])
            except Exception:
                cur_adx = None                       # can't measure -> don't block the trade
            if cur_adx is not None and cur_adx < min_adx:
                log(f"{symbol}: choppy market (ADX {cur_adx:.0f} < {min_adx}) -> skip")
                note(f"Skipped {_coin(symbol)} - the market is choppy, not trending "
                     f"(trend strength {cur_adx:.0f}, needs {min_adx}).")
                return
        best = pick_best_strategy(df, min_trades, r.get("max_backtest_return_pct", 300))
        if not best:
            log(f"{symbol}: no reliable+profitable strategy right now -> staying flat")
        else:
            name, fn, ret, ntr = best["name"], best["fn"], best["ret"], best["trades"]
            conf, need = best["confirmations"], r.get("min_confirmations", 3)
            parr = np.asarray(fn(df), float)
            sig = int(parr[-1])
            if sig == 0:
                log(f"{symbol}: best = {name} (+{ret:.0f}%, {ntr} trades) but signal Flat -> wait")
            elif conf < need:                          # require multi-strategy agreement
                log(f"{symbol}: {name} says {'LONG' if sig > 0 else 'SHORT'} but only {conf} "
                    f"strategies agree (need {need}) -> skip")
                note(f"Skipped {_coin(symbol)} - {name} wanted to go "
                     f"{'long' if sig > 0 else 'short'}, but only {conf} strategies agreed "
                     f"(needs at least {need}).")
            else:
                htf = _HIGHER_TF.get(timeframe, "1d")     # trend filter: trade only WITH the bigger trend
                try:
                    trend = higher_tf_trend(get_ohlcv(symbol, htf, 300, closed_only=True))
                except Exception:
                    trend = sig                            # can't fetch -> don't block
                if sig != trend:
                    log(f"{symbol}: {name} {'LONG' if sig > 0 else 'SHORT'} but {htf} trend is "
                        f"{'UP' if trend > 0 else 'DOWN'} -> skip (counter-trend)")
                    note(f"Skipped {_coin(symbol)} - {name} wanted to go "
                         f"{'long' if sig > 0 else 'short'}, but the bigger {htf} trend is "
                         f"{'up' if trend > 0 else 'down'}, so it would be fighting the trend.")
                    return
                leverage = auto_leverage(df, parr,     # normal trades keep their own (1x) cap,
                                         {**r, "max_leverage": r.get("normal_max_leverage", 1)})
                usdt = _fit_to_capital(t, usdt, r)     # 10% of wallet, capped to the free margin
                if usdt is None:
                    log(f"{symbol}: not enough free capital for even a minimum trade -> skip")
                    return
                guards.check_order(symbol, usdt, leverage)
                _, sl, _, slp = auto_tp_sl(df, sig, price, r, leverage)
                t.futures_market_order(symbol, "BUY" if sig > 0 else "SELL", usdt, leverage)
                stops = t.place_stops(symbol, sig, None, sl)   # hard SL only; trailing runs in the loop
                state[symbol] = {"side": sig, "entry": price, "sl": sl, "peak": price,
                                 "trail_pct": slp, "strategy": name, "timeframe": timeframe,
                                 "leverage": leverage, "size": usdt, "open_time": time.strftime("%Y-%m-%d %H:%M"),
                                 "expected_return": round(ret, 1), "confirmations": conf,
                                 "stop_placed": "placed" in str(stops)}
                save_state(state)
                log(f"OPEN {symbol} {'LONG' if sig > 0 else 'SHORT'} {usdt}USDT {leverage}x(auto) "
                    f"via {name} (+{ret:.0f}%, {ntr} trades, {conf} agree) | entry {price:.4f} "
                    f"SL {sl:.4f}(-{slp}%) | trailing {slp}% | stops: {stops}")
                note(f"Opened a {'LONG (betting up)' if sig > 0 else 'SHORT (betting down)'} trade on "
                     f"{_coin(symbol)} at {price:.6g} with {usdt} USDT. Chose the {name} strategy "
                     f"because it scored best in the backtest (+{ret:.0f}%), and {conf} strategies "
                     f"agreed on this direction. Safety stop-loss placed at {sl:.6g} ({slp}% away).")


def _open_news_trade(t, symbol, direction, usdt, r, state, headline):
    """Open a trade purely on a news signal (no strategy). Uses its own extra slot."""
    try:
        leverage = min(r.get("news_leverage", 2), r.get("max_leverage", 5))
        usdt = _fit_to_capital(t, usdt, r)                 # cap to free margin so the slot still fills
        if usdt is None:
            log(f"news trade {symbol}: not enough free capital -> skip")
            return False
        guards.check_order(symbol, usdt, leverage)
        price = t.futures_price(symbol)
        sl_pct = r.get("news_stop_loss_pct", 3.0)
        sl = price * (1 - sl_pct / 100) if direction > 0 else price * (1 + sl_pct / 100)
        t.futures_market_order(symbol, "BUY" if direction > 0 else "SELL", usdt, leverage)
        stops = t.place_stops(symbol, direction, None, sl)
        state[symbol] = {"side": direction, "entry": price, "sl": sl, "peak": price,
                         "trail_pct": sl_pct, "news": True, "headline": headline,
                         "leverage": leverage, "size": usdt, "timeframe": "4h",
                         "open_time": time.strftime("%Y-%m-%d %H:%M"),
                         "stop_placed": "placed" in str(stops)}
        save_state(state)
        log(f"NEWS TRADE {symbol} {'LONG' if direction > 0 else 'SHORT'} {usdt}USDT {leverage}x | {headline}")
        note(f"Opened a NEWS trade: {'LONG' if direction > 0 else 'SHORT'} on {_coin(symbol)} at "
             f"{price:.6g}, triggered by the headline: \"{headline}\". Stop-loss at {sl:.6g}.")
        return True
    except guards.RiskError as e:
        log(f"news trade blocked ({symbol}): {e}")
    except Exception as e:
        log(f"news trade error ({symbol}): {e}")
    return False


def _do_news_trades(t, r, state, usdt, seen, total_cap):
    """Check news; open news trades in their OWN slot(s), within the total position cap."""
    free = min(r.get("news_slots", 1) - sum(1 for v in state.values() if v.get("news")),
               total_cap - len(state))
    if free <= 0:
        return
    try:
        from news import get_news_signals
        signals = get_news_signals(max_age_min=r.get("news_max_age_min", 30),
                                   min_sentiment=r.get("news_min_sentiment", 1))
    except Exception as e:
        log(f"news check failed: {e}")
        return
    if len(seen) > 1000:
        seen.clear()
    for ns in signals:
        if free <= 0:
            break
        if ns["id"] in seen:
            continue
        seen.add(ns["id"])
        if ns["symbol"] in state:            # already trading this coin
            continue
        if LOSS_STREAK.get(ns["symbol"], 0) >= r.get("max_losses_per_coin", 2):
            continue                         # blocked after repeated losses
        if _on_cooldown(ns["symbol"], r):
            continue                         # re-entry cooldown active
        if _open_news_trade(t, ns["symbol"], ns["direction"], usdt, r, state, ns["headline"]):
            free -= 1


# ======================= SPECIAL (momentum / news) trades =======================
# A separate breed of trade with its OWN lifecycle (the normal strategy/trailing rules do
# NOT apply):
#   entry  : fresh strong news OR a volume spike + short-term momentum (see _special_candidates)
#   size   : same 10%-of-capital slot, but 5-10x leverage
#   stop   : a HARD 2% price stop placed as a REAL exchange order (~20% of the slot at 10x)
#   profit : at +special_target_pct% price move (=+100% margin @10x) -> bank 50%, let the rest run;
#            if that target isn't reached within special_deadline_hours -> bank 75%, let the rest run.
#            The runner is yours to close by hand (a safety max-hold still applies as a backstop).
# NOTE: "crypto chatrooms / Binance Square" is NOT wired in — there is no free live or historical
# feed for it here. News + volume-spike momentum are the available stand-ins; a Square/X source can
# be bolted on later exactly like news.py (see its header note).

def _record_partial(pos, symbol, exit_price, fraction, reason):
    """Log + record a partial profit-take (special trades bank 50%/75% and let the rest run)."""
    try:
        from history import record
        entry, side, lev = pos["entry"], pos["side"], pos.get("leverage", 1)
        move = (exit_price / entry - 1) * 100 * side
        pnl = round(move * lev, 2)
        note(f"Banked {fraction*100:.0f}% of the {_coin(symbol)} special trade at {exit_price:.6g} "
             f"- {pnl:+.2f}% on the part closed "
             f"({'target hit' if reason == 'target' else '24h deadline'}). "
             f"Letting the remaining {100 - fraction*100:.0f}% run.")
        record({
            "open_time": pos.get("open_time", ""),
            "close_time": time.strftime("%Y-%m-%d %H:%M"),
            "coin": symbol, "side": "LONG" if side > 0 else "SHORT",
            "type": "special-partial", "strategy": "special",
            "timeframe": pos.get("timeframe", "1h"),
            "leverage": lev, "confirmations": f"{fraction*100:.0f}%",
            "expected_return_%": pos.get("target_pct", ""),
            "entry": round(entry, 6), "exit": round(exit_price, 6),
            "price_move_%": round(move, 2), "pnl_%": pnl,
            "pnl_usdt": round(fraction * pos.get("size", 100) * pnl / 100, 2),  # only the % booked
            "size_usdt": pos.get("size", ""),                       # full margin used on this trade
            "exit_reason": f"partial-{reason}",
        })
    except Exception as e:
        log(f"partial record failed ({symbol}): {e}")


def _special_trigger(sym, r):
    """Return (direction, reason) if a special ENTRY trigger fires, else None.
    The backtested 'recommended (1,2,3,4)' triggers, evaluated on the LATEST candles (OR of):
      (4) volume ACCELERATION — 3 rising 1h candles + a moderate spike
      (3) range BREAKOUT — a tight multi-hour consolidation then a break on volume (earliest entry)
      (1) 15m volume SPIKE — a fast burst on the 15m timeframe
    Direction = short-term momentum (breakouts trade with the break)."""
    try:
        h1 = get_ohlcv(sym, "1h", 30, closed_only=True, fut_first=True)   # futures perp -> 1 request
    except Exception:
        return None
    if h1 is None or len(h1) < 14:
        return None
    c = h1["close"].values; v = h1["volume"].values; hi = h1["high"].values; lo = h1["low"].values
    base = v[-6:-1].mean()
    mom = float(c[-1] / c[-4] - 1)                              # ~3h momentum
    # (4) volume acceleration: three rising candles + a moderate spike (catches the ignition ramp)
    if v[-1] > v[-2] > v[-3] and base > 0 and v[-1] >= r.get("special_accel_spike", 1.5) * base:
        return (1 if mom >= 0 else -1, f"volume accelerating x{v[-1]/base:.1f}, {mom*100:+.1f}% 3h")
    # (3) range breakout: tight consolidation over K hours, then a break on above-average volume
    K = int(r.get("special_range_hours", 12))
    if len(h1) > K + 1:
        rh = hi[-K-1:-1].max(); rl = lo[-K-1:-1].min()
        vbase = v[-K-1:-1].mean()
        width = (rh - rl) / rl if rl > 0 else 9.0
        if width <= r.get("special_range_max_width", 0.08) and vbase > 0 and v[-1] >= 1.5 * vbase:
            if c[-1] > rh:
                return (1, f"range breakout +{(c[-1]/rh-1)*100:.1f}% over {K}h base")
            if c[-1] < rl:
                return (-1, f"range breakdown {(c[-1]/rl-1)*100:.1f}% under {K}h base")
    # (1) 15m volume spike: a fast burst before the 1h candle would confirm
    try:
        m = get_ohlcv(sym, "15m", 30, closed_only=True, fut_first=True)
        v5 = m["volume"].values; c5 = m["close"].values
        b5 = v5[-6:-1].mean()
        if b5 > 0 and v5[-1] / b5 >= r.get("special_vol_spike", 2.0):
            mm = float(c5[-1] / c5[-4] - 1)
            return (1 if mm >= 0 else -1, f"15m volume spike x{v5[-1]/b5:.1f}, {mm*100:+.1f}% 45m")
    except Exception:
        pass
    return None


def _oi_surge(sym, r):
    """OPEN-INTEREST gate: True only if OI (new money) rose >= threshold over the last N hours.
    This is the single biggest edge in the backtest — it separates real, funded pumps from thin
    spot-only spikes (which slipped and lost). If OI can't be measured, reject when it's required."""
    try:
        from data import open_interest_change
        ch = open_interest_change(sym, hours=int(r.get("special_oi_hours", 4)))
    except Exception:
        ch = None
    if ch is None:
        return not r.get("special_oi_required", True)
    return ch >= r.get("special_oi_surge_pct", 3.0) / 100.0


def _special_candidates(r, seen, state):
    """Coins likely to run: OI-confirmed momentum + strong fresh news.
    Entry filter = the backtested 'recommended (1,2,3,4)' config — three momentum triggers
    (15m spike / range breakout / volume acceleration), UNIONED, then gated by an OPEN-INTEREST
    surge (new money) and the >=50M LIQUIDITY filter. News keeps its own high-signal path."""
    from concurrent.futures import ThreadPoolExecutor
    out, picked = [], set()
    try:
        from data import quote_volumes
        liq = quote_volumes()
    except Exception:
        liq = {}
    min_qv = r.get("special_min_volume", 50_000_000)     # only trade coins >= this 24h $volume
    exclude = set(r.get("special_exclude", []))          # too-big/too-slow coins never worth a special slot
    # 1) news-driven (its own high-signal source; not OI-gated — news often leads the OI move)
    try:
        from news import get_news_signals
        for ns in get_news_signals(max_age_min=r.get("special_news_max_age_min", 60),
                                   min_sentiment=r.get("special_news_min_sentiment", 2)):
            if ns["id"] in seen or ns["symbol"] in picked or ns["symbol"] in exclude:
                continue
            seen.add(ns["id"]); picked.add(ns["symbol"])
            out.append({"symbol": ns["symbol"], "direction": ns["direction"],
                        "reason": f"news: {ns['headline']}"})
    except Exception as e:
        log(f"special news scan failed: {e}")
    # 2) OI-gated momentum triggers across the whole >=50M liquid universe (parallel scan)
    if liq:
        universe = [s for s, vol in liq.items() if vol >= min_qv]
    else:
        try:
            from data import high_volatility_coins
            universe = high_volatility_coins(n=r.get("special_scan_movers", 25), min_quote_volume=min_qv)
        except Exception:
            universe = []
    universe = [s for s in universe if s not in picked and s not in state and s not in exclude]

    def _eval(sym):
        trig = _special_trigger(sym, r)                  # (direction, reason) or None
        if not trig:
            return None
        if not _oi_surge(sym, r):                        # OI gate: require new money
            return None
        return {"symbol": sym, "direction": trig[0], "reason": trig[1]}
    try:
        with ThreadPoolExecutor(max_workers=16) as ex:   # scan ~44 coins fast, in parallel
            for res in ex.map(_eval, universe):
                if res and res["symbol"] not in picked:
                    picked.add(res["symbol"]); out.append(res)
    except Exception as e:
        log(f"special momentum scan failed: {e}")
    # LIQUIDITY FILTER + skip cooldown / repeated losers. Empty liq map -> don't block.
    return [o for o in out
            if (not liq or liq.get(o["symbol"], 0) >= min_qv)
            and LOSS_STREAK.get(o["symbol"], 0) < r.get("max_losses_per_coin", 2)
            and not _on_cooldown(o["symbol"], r)]


def _special_leverage(symbol, r):
    """Auto-pick leverage in the 5-10x band from the coin's recent hourly volatility:
    a calm/liquid coin gets the max (10x); a jumpy/thin coin gets the min (5x), so a 2% stop
    never erodes more than intended. Falls back to the MIN (safest) if it can't measure."""
    lo = int(r.get("special_leverage_min", 5))
    hi = int(r.get("special_leverage_max", 10))
    try:
        df = get_ohlcv(symbol, "1h", 48, closed_only=True)
        vol = float(df["close"].pct_change().tail(24).std() * 100)   # % hourly volatility
    except Exception:
        return lo
    calm = r.get("special_vol_calm_pct", 1.0)      # <= this % -> max leverage (calm coin)
    wild = r.get("special_vol_wild_pct", 3.0)      # >= this % -> min leverage (jumpy coin)
    if vol <= calm:
        return hi
    if vol >= wild:
        return lo
    frac = (vol - calm) / (wild - calm)                              # scale linearly in between
    return int(round(hi - frac * (hi - lo)))


def _open_special_trade(t, symbol, direction, usdt, r, state, reason):
    try:
        lev = _special_leverage(symbol, r)                           # auto 5-10x by coin volatility
        usdt = _fit_to_capital(t, usdt, r)                           # cap to free margin (fills the 2nd slot)
        if usdt is None:
            log(f"special {symbol}: not enough free capital -> skip")
            return False
        guards.check_order(symbol, usdt, lev)
        price = t.futures_price(symbol)
        # MARGIN-based: stop = -20% of margin, target = +100% of margin, at ANY leverage.
        # In price terms that's (margin% / leverage): e.g. 20%/5x = 4% price, 100%/5x = 20% price.
        sl_pct = r.get("special_stop_margin_pct", 20.0) / lev        # price move for a -20% margin loss
        tgt_pct = r.get("special_target_margin_pct", 100.0) / lev    # price move for a +100% margin gain
        sl = price * (1 - sl_pct / 100) if direction > 0 else price * (1 + sl_pct / 100)
        t.futures_market_order(symbol, "BUY" if direction > 0 else "SELL", usdt, lev)
        stops = t.place_stops(symbol, direction, None, sl)          # hard stop = -20% of margin, on the exchange
        state[symbol] = {"side": direction, "entry": price, "sl": sl, "peak": price,
                         "special": True, "leverage": lev, "size": usdt, "timeframe": "1h",
                         "target_pct": tgt_pct,
                         "open_epoch": time.time(), "open_time": time.strftime("%Y-%m-%d %H:%M"),
                         "reason": reason[:160], "partial_done": False,
                         "stop_placed": "placed" in str(stops)}
        save_state(state)
        log(f"SPECIAL {symbol} {'LONG' if direction > 0 else 'SHORT'} {usdt}USDT {lev}x | "
            f"stop -{sl_pct:.1f}%px (=-20% margin) @ {sl:.6g} | target +{tgt_pct:.0f}%px (=+100% margin) | {reason[:60]}")
        note(f"Opened a SPECIAL {'LONG' if direction > 0 else 'SHORT'} trade on {_coin(symbol)} at "
             f"{price:.6g} ({lev}x), because - {reason}. Hard stop at {sl:.6g} "
             f"({sl_pct:.1f}% away = 20% of the margin). Will bank 50% at +{tgt_pct:.0f}% price "
             f"(= +100% of the margin) and let the rest run.")
        return True
    except guards.RiskError as e:
        log(f"special trade blocked ({symbol}): {e}")
    except Exception as e:
        log(f"special trade error ({symbol}): {e}")
    return False


def _do_special_trades(t, r, state, usdt, seen, total_cap):
    """Fill the special slot(s) from news / volume-spike momentum, within the total cap."""
    free = min(r.get("special_slots", 2) - sum(1 for v in state.values() if v.get("special")),
               total_cap - len(state))
    if free <= 0:
        return
    if len(seen) > 1000:
        seen.clear()
    for c in _special_candidates(r, seen, state):
        if free <= 0:
            break
        if c["symbol"] in state:                 # never double-up on a coin
            continue
        if _open_special_trade(t, c["symbol"], c["direction"], usdt, r, state, c["reason"]):
            free -= 1


def _special_breakeven(t, pos, symbol, state):
    """After banking the target, move the hard stop to entry so the runner can't become a loss."""
    try:
        t.cancel_open_orders(symbol)
        t.place_stops(symbol, pos["side"], None, pos["entry"])
        pos["sl"] = pos["entry"]
        save_state(state)
    except Exception as e:
        log(f"special breakeven ({symbol}) failed: {e}")


def _ladder_floor(peak_margin, r):
    """Wide profit-lock ladder (all in MARGIN %): the runner's peak margin -> a locked floor.
    Monotonic, configurable. Default: peak +80% -> lock +20%, +150% -> +50%, +250% -> +120%,
    +400% -> +250%. Wide on purpose, so a real pump keeps running instead of being trimmed early."""
    steps = r.get("special_runner_ladder", [[80, 20], [150, 50], [250, 120], [400, 250]])
    floor = None
    for th, fl in steps:
        if peak_margin >= th:
            floor = fl
    return floor


def _special_runner_trail(t, symbol, pos, price, r, state):
    """C: AFTER the +100% target is banked (50% off + breakeven), let the RUNNER ride the WIDE ladder
    — a locked floor that GROWS with the peak margin. Done IN-PROCESS: the bot closes the runner at
    MARKET the moment the live price falls back through the current floor. This replaces the old
    approach of moving the exchange stop by cancel-then-place, which kept failing with -4130 ('a
    close-position order already exists in this direction') so the lock never advanced (the ONUSDT
    give-back). The breakeven exchange stop stays as a backstop for when the bot is offline."""
    if not pos.get("partial_done"):
        return                                            # runner only — before the +100% bank, hard stop rules
    side, entry, lev = pos["side"], pos["entry"], pos.get("leverage", 1)
    peak = pos.get("peak", entry)
    peak_margin = (peak / entry - 1) * 100 * side * lev
    floor = _ladder_floor(peak_margin, r)
    if floor is None:
        return                                            # not high enough on the ladder yet (peak < +80%)
    floor_price = entry * (1 + floor / lev / 100.0) if side > 0 else entry * (1 - floor / lev / 100.0)
    if pos.get("trail_locked_margin") != round(floor, 1):
        pos["trail_locked_margin"] = round(floor, 1)      # ratcheted up a rung -> record + show it
        save_state(state)
        log(f"SPECIAL {symbol}: runner lock raised to +{floor:.0f}% margin (peak +{peak_margin:.0f}%)")
        note(f"Locked in +{floor:.0f}% of the margin on the {_coin(symbol)} runner "
             f"(it has peaked at +{peak_margin:.0f}%).")
    # price fell back through the floor? -> bank the runner at market (no exchange-stop juggling, no -4130)
    crossed = (side > 0 and price <= floor_price) or (side < 0 and price >= floor_price)
    if crossed:
        try:
            _record_close(pos, symbol, price, f"runner-lock +{floor:.0f}%")
            t.close_futures(symbol)
            t.cancel_open_orders(symbol)
            state.pop(symbol, None)
            save_state(state)
            log(f"SPECIAL CLOSE {symbol} @ {price:.6g} (runner banked +{floor:.0f}% margin, "
                f"peak was +{peak_margin:.0f}%)")
        except Exception as e:
            log(f"special runner close ({symbol}) failed: {e}")


def _special_take_target(t, symbol, pos, price, r, state):
    """Bank the target-partial (50%) + move to breakeven the INSTANT a special trade reaches
    +target%. Takes the price as an argument so the 1s watcher can call it with no extra API
    round-trip — this is what stops the +10% window being missed. Returns True if it acted."""
    if pos.get("partial_done"):
        return False
    move = (price / pos["entry"] - 1) * 100 * pos["side"]
    if move < pos.get("target_pct", r.get("special_target_pct", 10.0)):
        return False
    frac = r.get("special_partial_at_target", 0.50)
    t.partial_close(symbol, frac)
    _record_partial(pos, symbol, price, frac, "target")
    pos["partial_done"] = True
    pos["partial_reason"] = "target"
    pos["remaining_frac"] = round(1.0 - frac, 4)          # only this fraction is still open (for correct runner $)
    save_state(state)
    if r.get("special_breakeven_after_partial", True):
        _special_breakeven(t, pos, symbol, state)
    log(f"SPECIAL {symbol}: +{move:.1f}% target hit -> banked {frac*100:.0f}%, letting the rest run")
    return True


def _manage_special(t, symbol, r, state):
    """One special-trade cycle: partial-TP at the target, timed partial at the deadline, then let
    the runner go until its (breakeven / 2%) exchange stop or a safety max-hold."""
    pos = state.get(symbol)
    if not pos:
        return
    side, entry, lev = pos["side"], pos["entry"], pos.get("leverage", 1)
    try:
        price = t.futures_price(symbol)
    except Exception as e:
        log(f"special {symbol}: price fetch failed ({e})")
        return
    # exchange may have closed us (2% stop / breakeven stop filled, or a manual close)
    live = {p["symbol"] for p in t.futures_positions()}
    if symbol not in live:
        _record_close(pos, symbol, price, "closed on exchange")
        t.cancel_open_orders(symbol)
        state.pop(symbol, None)
        save_state(state)
        log(f"SPECIAL CLOSE {symbol} @ {price:.6g} (stop filled / closed by hand)")
        return

    move = (price / entry - 1) * 100 * side           # price move in our favour (%)
    pk = pos.get("peak", entry)
    pos["peak"] = max(pk, price) if side > 0 else min(pk, price)     # keep the high-water mark fresh
    age_h = (time.time() - pos.get("open_epoch", time.time())) / 3600.0
    deadline = r.get("special_deadline_hours", 24)

    if not pos.get("partial_done"):
        if _special_take_target(t, symbol, pos, price, r, state):   # +target% -> bank 50% + breakeven
            return
        # 24h deadline bank-75% is OFF by default: the 90-day backtest showed it prematurely cashes
        # out would-be pumps and turned C from +$3.4k into -$6.7k. Re-enable via special_deadline_enabled.
        if r.get("special_deadline_enabled", False) and age_h >= deadline:
            frac = r.get("special_partial_at_deadline", 0.75)
            t.partial_close(symbol, frac)
            _record_partial(pos, symbol, price, frac, "deadline")
            pos["partial_done"] = True
            pos["partial_reason"] = "deadline"
            save_state(state)
            log(f"SPECIAL {symbol}: {deadline}h passed at {move:+.1f}% -> banked "
                f"{frac*100:.0f}%, letting the rest run")
            return

    # runner: trail on the WIDE ladder (in-process), then hold until it / a manual close / max-hold
    if pos.get("partial_done"):
        _special_runner_trail(t, symbol, pos, price, r, state)
    max_hold = r.get("special_runner_max_hold_hours", 72)
    if pos.get("partial_done") and age_h >= max_hold:
        _record_close(pos, symbol, price, "runner max-hold")
        t.close_futures(symbol)
        t.cancel_open_orders(symbol)
        state.pop(symbol, None)
        save_state(state)
        log(f"SPECIAL CLOSE {symbol} @ {price:.6g} (runner hit {max_hold}h safety max-hold)")
        return
    tag = "LONG" if side > 0 else "SHORT"
    stage = "runner" if pos.get("partial_done") else "pre-target"
    log(f"HOLD SPECIAL {symbol} {tag} @ {price:.6g} | {move:+.1f}% price ({move*lev:+.0f}% margin) "
        f"| {stage} | age {age_h:.1f}h | stop {pos['sl']:.6g}")


def run_auto(capital, max_coins=None):
    """Fully automatic multi-coin mode: auto-pick best coins + timeframe, trade up to N at once,
    plus a separate extra slot for news-driven trades."""
    from autobot import scan_and_rank, UNIVERSE
    t = BinanceTrader()
    try:                                               # keep bot.pid == the live process so the
        open("bot.pid", "w").write(str(os.getpid()))   # dashboard shows 'running' (bot is a service now)
    except Exception:
        pass
    r = rules()
    interval = r.get("check_interval_seconds", 300)
    total_cap = max_coins or r.get("max_concurrent_positions", 10)
    reg_slots = r.get("regular_slots", 7)
    news_slots = r.get("news_slots", 1)
    spec_slots = r.get("special_slots", 2)
    pct = r.get("trade_pct_of_capital", 0.10)
    try:                                               # show the REAL wallet balance, not the CLI fallback
        start_bal = t.futures_balance().get("USDT", 0) or capital
    except Exception:
        start_bal = capital
    log(f"AUTO BOT START | balance {start_bal:.0f} USDT | {pct*100:.0f}% per trade (~{start_bal*pct:.0f} USDT) | "
        f"{total_cap} slots = {reg_slots} normal + {news_slots} news + {spec_slots} special | "
        f"{'TESTNET (fake money)' if t.testnet else 'LIVE (REAL money)'} | check every {interval}s")
    note(f"--- Bot started. Wallet balance {start_bal:.0f} USDT, using {pct*100:.0f}% per trade "
         f"(~{start_bal*pct:.0f} USDT each), {total_cap} slots "
         f"total: {reg_slots} normal strategy trades, {news_slots} news trade, {spec_slots} special "
         f"momentum trades (5-10x, OI-confirmed entries, stop -20% of margin, bank 50% at +100% of "
         f"margin, then ride the runner on a wide profit-lock ladder). "
         f"On {'TESTNET (fake money)' if t.testnet else 'LIVE money'}. "
         f"Only trading normal when trend strength (ADX) is at least {r.get('min_adx', 0)}. "
         f"Watching open trades every {r.get('watch_interval_seconds', 1)}s, "
         f"looking for new trades every {interval}s. ---")
    state = load_state()
    seen_news = set()
    seen_special = set()

    while True:
        try:
            r = rules()
            coins = r.get("allowed_symbols") or None       # empty list -> dynamic all-coins scan
            # CAPITAL = the LIVE futures wallet balance (works for demo AND live). Each trade is a
            # fresh 10% of the CURRENT balance, so the size shrinks/grows with the account.
            try:
                live_bal = t.futures_balance().get("USDT", 0)
            except Exception:
                live_bal = 0
            cap_now = live_bal or capital                  # fall back to the CLI value if the fetch fails
            trade_usdt = round(min(cap_now * r.get("trade_pct_of_capital", 0.10),
                                   r.get("max_usdt_per_trade", 500)), 2)
            # 1) manage every open position (regular + news) — isolate each coin
            for sym in list(state.keys()):
                try:
                    step(t, sym, trade_usdt, r, state, state[sym].get("timeframe", "4h"))
                except Exception as e:
                    log(f"manage {sym} skipped this cycle: {e}")
            # 2) fill NORMAL strategy slots (news + special use their own pools, not these)
            regular = sum(1 for v in state.values() if not v.get("news") and not v.get("special"))
            slots = min(reg_slots - regular, total_cap - len(state))
            if slots > 0:
                log(f"Scanning for the best setups… ({slots} normal slot(s) free)")
                picks = scan_and_rank(coins, min_trades=r.get("min_trades_for_strategy", 20),
                                      coin_limit=r.get("scan_coin_limit", 120),
                                      max_return=r.get("max_backtest_return_pct", 300),
                                      spike_low=r.get("volume_spike_low", 1.5),
                                      spike_high=r.get("volume_spike_high", 2.0),
                                      spike_boost=r.get("volume_spike_boost", 1.5),
                                      spike_exclude=r.get("volume_spike_exclude"),
                                      min_adx=r.get("min_adx", 0))
                save_recommendations(picks)          # publish live suggestions for the dashboard
                for pick in picks:
                    if slots <= 0:
                        break
                    if not _can_open_now(r):              # spread entries out over time
                        log(f"batch limit: {r.get('max_trades_per_batch', 3)} trades opened in the last "
                            f"{r.get('open_batch_window_min', 30)} min -> waiting for better entries")
                        break
                    if pick["symbol"] in state or pick["signal"] == 0:
                        continue
                    try:
                        step(t, pick["symbol"], trade_usdt, r, state, pick["timeframe"])
                    except Exception as e:
                        log(f"open {pick['symbol']} skipped: {e}")
                        continue
                    if pick["symbol"] in state:
                        slots -= 1
                        RECENT_OPENS.append(time.time())
            # 3) SPECIAL trades — momentum/news, 5-10x, own 2% stop + timed partial TP
            if r.get("special_enabled", True):
                try:
                    _do_special_trades(t, r, state, trade_usdt, seen_special, total_cap)
                except Exception as e:
                    log(f"special check failed: {e}")
            # 4) NEWS trades — their own slot, within the total cap
            if r.get("news_enabled", True):
                _do_news_trades(t, r, state, trade_usdt, seen_news, total_cap)
        except guards.RiskError as e:
            log(f"BLOCKED by safety rule: {e}")
        except Exception as e:
            log(f"auto error (will retry): {e}")
        # ---- FAST WATCH: until the next scan is due, check live prices every second ----
        fast = max(0.5, float(r.get("watch_interval_seconds", 1)))
        deadline = time.time() + interval
        while time.time() < deadline:
            t0 = time.time()
            try:
                watch_positions(t, r, state)
            except Exception as e:
                log(f"fast watch error: {e}")
            time.sleep(max(0.2, fast - (time.time() - t0)))   # keep the true cycle at ~fast seconds


def run_bot(symbol, usdt, timeframe=TIMEFRAME):
    symbol = symbol.upper()
    t = BinanceTrader()
    interval = rules().get("check_interval_seconds", 300)
    leverage = rules().get("default_leverage", 2)

    guards.check_order(symbol, usdt, leverage)  # fail fast if the request itself is unsafe
    log(f"BOT START | {symbol} | {usdt} USDT | {leverage}x | {timeframe} | "
        f"{'TESTNET (fake money)' if t.testnet else 'LIVE (REAL money)'} | check every {interval}s")
    state = load_state()

    while True:
        try:
            step(t, symbol, usdt, rules(), state, timeframe)
        except guards.RiskError as e:
            log(f"BLOCKED by safety rule: {e}")
        except Exception as e:
            log(f"error (will retry): {e}")
        time.sleep(interval)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:\n"
              "  python bot.py <SYMBOL> <USDT> [TIMEFRAME]   # one coin, e.g. python bot.py ETHUSDT 50 4h\n"
              "  python bot.py auto <CAPITAL> [MAX_COINS]     # fully automatic, e.g. python bot.py auto 1000 5")
        sys.exit(1)
    try:
        if sys.argv[1].lower() == "auto":
            if len(sys.argv) > 3:
                mc = int(sys.argv[3])
            else:                                   # no CLI max_coins -> use the dashboard's shared setting
                try:
                    mc = int(json.load(open("ui_settings.json")).get("max_coins", 10))
                except Exception:
                    mc = 10
            run_auto(float(sys.argv[2]), mc)
        else:
            run_bot(sys.argv[1], float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else TIMEFRAME)
    except KeyboardInterrupt:
        print("\nBot stopped by user. Open positions (if any) remain — close them in the UI or the testnet site.")
