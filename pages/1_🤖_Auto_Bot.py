"""Auto Bot — clean live dashboard: status, open positions, auto suggestions,
activity feed, performance. Suggestions auto-update from the bot's own scan (no button)."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
PID, LOG, STATE, HIST, RECS, DIARY = (ROOT / f for f in
                                      ("bot.pid", "bot.log", "bot_state.json", "trade_history.csv",
                                       "recommendations.json", "activity.log"))

st.set_page_config(page_title="RL - Trade bot", page_icon="🤖", layout="wide")
st.markdown("""<style>
.scard{background:#f5f7fa;border:1px solid #e6e9ef;border-radius:12px;padding:.7rem .9rem}
.scard .lbl{color:#667085;font-size:12px}.scard .val{font-size:20px;font-weight:500;margin-top:.1rem}
.poscard{background:#f5f7fa;border:1px solid #e6e9ef;border-radius:12px;padding:.7rem .9rem;margin-bottom:.5rem}
.badge{font-size:11px;padding:.1rem .45rem;border-radius:20px;font-weight:500}
.sub{color:#667085;font-size:12px;margin-top:.2rem}
.track{position:relative;height:6px;background:#e6e9ef;border-radius:20px;margin-top:.5rem}
.eline{position:absolute;top:-3px;width:2px;height:12px;background:#98a2b3}
.dot{position:absolute;top:-3px;width:12px;height:12px;border-radius:50%;transform:translateX(-50%)}
.feed{background:#f5f7fa;border:1px solid #e6e9ef;border-radius:12px;padding:.2rem .9rem}
.fitem{display:flex;gap:.55rem;align-items:baseline;padding:.35rem 0;border-bottom:1px solid #e6e9ef;font-size:14px}
.fitem:last-child{border-bottom:none}.ft{color:#667085;font-size:12px;min-width:44px}
.h{font-size:16px;font-weight:500;margin:1.2rem 0 .5rem}
.hwrap{max-height:460px;overflow:auto;background:#fff;border:1px solid #e6e9ef;border-radius:12px}
.htbl{width:100%;border-collapse:collapse;font-size:13px}
.htbl th{position:sticky;top:0;background:#f5f7fa;text-align:left;color:#667085;font-weight:500;
         font-size:11px;padding:.5rem .6rem;border-bottom:1px solid #e6e9ef;white-space:nowrap}
.htbl td{padding:.45rem .6rem;border-bottom:1px solid #eef1f6;white-space:nowrap}
.htbl tr:last-child td{border-bottom:none}
.rsub{color:#98a2b3;font-size:11px}
</style>""", unsafe_allow_html=True)

st.title("🤖 RL - Trade bot")

# ---------- process helpers (cross-platform) ----------
def running_pid():
    if not PID.exists():
        return None
    pid = PID.read_text().strip()
    try:
        n = int(pid)
    except ValueError:
        return None
    if os.name == "nt":
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {n}"], capture_output=True, text=True).stdout
            return pid if str(n) in out else None
        except Exception:
            return pid
    try:
        os.kill(n, 0)
        return pid
    except OSError:
        return None


def stop_pid(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    else:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass


IS_SVC = os.name != "nt"                        # non-Windows host (systemd fallback)
CONTROL = os.getenv("RL_CONTROL", "").lower()   # "docker" when running in the container stack


def _docker_bot(action):
    """Control the rl-bot CONTAINER via the mounted Docker socket. action = start / stop / status."""
    try:
        import docker as _d
        c = _d.from_env().containers.get("rl-bot")
        if action == "start":
            c.start()
        elif action == "stop":
            c.stop(timeout=15)
        c.reload()
        return c.status
    except Exception:
        return None


def bot_is_running():
    """True if the bot is running. Docker: the rl-bot container is up. VM (systemd): the rl-bot
    service. Local: the pid file."""
    if CONTROL == "docker":
        return _docker_bot("status") == "running"
    if IS_SVC:
        try:
            out = subprocess.run(["systemctl", "is-active", "rl-bot"],
                                 capture_output=True, text=True).stdout.strip()
            return out == "active"
        except Exception:
            pass
    return running_pid() is not None


# ---------- password gate ----------
PIN = os.getenv("BOT_UI_PIN", "")
can_trade = True
if PIN:
    entered = st.sidebar.text_input("🔒 Trading PIN", type="password")
    can_trade = entered == PIN
    if can_trade:
        st.sidebar.success("Trading unlocked")
    else:
        st.sidebar.info("Enter PIN to enable Start/Stop")
else:
    st.sidebar.warning("No BOT_UI_PIN in .env — set one before using ngrok!")

# ---------- controls (settings persist to a shared file so EVERY user sees the same values) ----------
SETTINGS = ROOT / "ui_settings.json"
try:
    _cfg = json.load(open(SETTINGS))
except Exception:
    _cfg = {}
c2, c3, c4 = st.columns([2, 1, 1])
max_coins = c2.slider("🪙 Max coins", 1, 10, int(_cfg.get("max_coins", 5)))
if int(max_coins) != _cfg.get("max_coins"):
    try:                                    # save any change so all users stay in sync (last change wins)
        json.dump({"max_coins": int(max_coins)}, open(SETTINGS, "w"))
    except Exception:
        pass
running = bot_is_running()
if can_trade and not running and c3.button("▶ Start", type="primary", use_container_width=True):
    if CONTROL == "docker":                 # start the rl-bot container (latest image, on the shared code)
        _docker_bot("start")
    elif IS_SVC:                            # start the auto-restart systemd service
        subprocess.run(["sudo", "systemctl", "start", "rl-bot"], capture_output=True)
    else:                                  # local dev fallback (capital comes from live balance in-bot)
        with open(LOG, "a") as f:
            subprocess.Popen([sys.executable, "bot.py", "auto", "1000"],
                             cwd=str(ROOT), stdout=f, stderr=f)
    st.rerun()
if running and c4.button("⏹ Stop", use_container_width=True):
    if CONTROL == "docker":
        _docker_bot("stop")
    elif IS_SVC:
        subprocess.run(["sudo", "systemctl", "stop", "rl-bot"], capture_output=True)
    else:
        p = running_pid()
        if p:
            stop_pid(p)
        PID.unlink(missing_ok=True)
    st.rerun()


def esc(x):
    return str(x)


def fnum(v):
    try:
        v = float(v)
        return f"{v:.6g}"
    except Exception:
        return "—"


# ---------- live dashboard (auto-refresh) ----------
@st.fragment(run_every=20)
def dashboard():
    run = bot_is_running()
    # ---- gather data ----
    positions, balance, testnet, err = [], None, True, None
    try:
        from trader.client import BinanceTrader
        t = BinanceTrader()
        testnet = t.testnet
        balance = t.futures_balance().get("USDT")
        positions = t.futures_positions()
    except Exception as e:
        err = str(e)
    state = {}
    if STATE.exists():
        try:
            state = json.load(open(STATE))
        except Exception:
            pass
    rules = {}
    try:
        rules = json.load(open(ROOT / "risk_rules.json"))
    except Exception:
        pass
    t_act = rules.get("trail_activation_pct", 2.0)          # trailing arms at +this%
    t_tighten = rules.get("trail_tighten_at_pct", 5.0)      # tightens past +this%
    t_norm = rules.get("trail_pct_normal", 2.0)
    t_tight = rules.get("trail_pct_tight", 1.2)
    hist = None
    if HIST.exists():
        try:
            hist = pd.read_csv(HIST)
            hist["pnl"] = pd.to_numeric(hist["pnl_%"], errors="coerce")
            hist = hist.dropna(subset=["pnl"])
        except Exception:
            hist = None

    # ---- status bar ----
    dot = "#178a3a" if run else "#98a2b3"
    status = "Bot running" if run else "Bot stopped"
    mode = "testnet · 1x" if testnet else "LIVE · 1x"
    win = f"{(hist['pnl'] > 0).mean() * 100:.0f}%" if hist is not None and len(hist) else "—"
    bal = f"${balance:,.0f}" if balance is not None else "—"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:.6rem;margin:.4rem 0 1rem">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{dot};display:inline-block"></span>'
        f'<span style="font-size:16px;font-weight:500">{status}</span>'
        f'<span class="badge" style="background:#eef1f6;color:#667085">{mode}</span></div>',
        unsafe_allow_html=True)
    s = st.columns(4)
    for col, lbl, val in zip(s, ["Balance", "Open trades", "Win rate", "Closed trades"],
                             [bal, f"{len(positions)} / {max_coins}", win,
                              str(len(hist)) if hist is not None else "0"]):
        col.markdown(f'<div class="scard"><div class="lbl">{lbl}</div><div class="val">{val}</div></div>',
                     unsafe_allow_html=True)

    # ---- open positions ----
    st.markdown('<div class="h">📌 Open positions</div>', unsafe_allow_html=True)
    if not positions:
        st.markdown('<div class="sub">No open trades right now.</div>', unsafe_allow_html=True)
    for p in positions:
        sym, amt, entry, upnl = p["symbol"], p["amount"], p["entry"], p["pnl"]
        side = 1 if amt > 0 else -1
        mark = entry + upnl / amt if amt else entry
        pl = (mark / entry - 1) * 100 * side if entry else 0
        stt = state.get(sym, {})
        sl = stt.get("sl", entry * (1 - 0.20 * side))
        strat = stt.get("strategy", "—")
        lev = stt.get("leverage", 1)
        peak = stt.get("peak", entry)
        peak = max(peak, mark) if side > 0 else min(peak, mark)   # include latest tick
        peak_profit = (peak / entry - 1) * 100 * side if entry else 0
        col = "#178a3a" if pl >= 0 else "#c0392b"
        # ---- middle field: SPECIAL trades show target/partial; everything else shows trailing ----
        if stt.get("special"):
            sl_pct = abs((sl / entry - 1) * 100) if entry else 0
            sl_txt = f'{fnum(sl)} (-{sl_pct:.0f}%)'
            if stt.get("partial_done"):
                banked = "50%" if stt.get("partial_reason") == "target" else "75%"
                lk = stt.get("trail_locked_margin")
                if lk:
                    mid = f'banked {banked} &middot; 🔒 runner locked +{lk:g}% margin'
                else:
                    mid = f'banked {banked} &middot; runner riding the ladder'
            else:
                mid = f'target +{stt.get("target_pct", 10):g}% &rarr; bank 50%'
        else:
            sl_txt = fnum(sl)
            if peak_profit >= t_act:                              # trailing ARMED -> open lock 🔓
                tpct = t_tight if peak_profit >= t_tighten else t_norm
                tstop = peak * (1 - tpct / 100) if side > 0 else peak * (1 + tpct / 100)
                locked = (tstop / entry - 1) * 100 * side         # profit locked in
                lcol = "#178a3a" if locked >= 0 else "#c0392b"
                mid = (f'🔓 trail {tpct:g}% @ {fnum(tstop)} &rarr; '
                       f'<span style="color:{lcol}">{locked:+.1f}% locked</span>')
            else:                                                 # NOT armed -> closed lock 🔒
                mid = f'🔒 trail off (arms +{t_act:g}%)'
        # ---- header tag by trade type ----
        if stt.get("special"):
            tag_html = '<span class="sub" style="margin:0;color:#7c5cff">⚡ special</span>'
        elif stt.get("news"):
            tag_html = '<span class="sub" style="margin:0">📰 news</span>'
        else:
            tag_html = f'<span class="sub" style="margin:0">{strat}</span>'
        bcol = "#178a3a" if side > 0 else "#c0392b"
        bbg = "rgba(23,138,58,.12)" if side > 0 else "rgba(192,57,43,.12)"
        arrow, txt = ("▲", "long") if side > 0 else ("▼", "short")
        posp = max(0, min(100, (pl + 20) / 30 * 100))
        card = (
            f'<div class="poscard"><div style="display:flex;align-items:center;gap:.5rem">'
            f'<span class="badge" style="background:{bbg};color:{bcol}">{arrow} {txt}</span>'
            f'<span style="font-weight:500">{sym}</span>{tag_html}'
            f'<span style="margin-left:auto;text-align:right;line-height:1.2">'
            f'<span style="display:block;font-weight:500;color:{col};font-size:17px">{pl*lev:+.1f}%</span>'
            f'<span style="display:block;color:{col};font-size:13px">{upnl:+.2f} USDT</span>'
            f'<span class="sub" style="display:block;margin:0;font-size:11px">price {pl:+.1f}%</span></span></div>'
            f'<div class="sub">enter {fnum(entry)} &middot; now {fnum(mark)} &middot; stoploss {sl_txt} '
            f'&middot; {mid} &middot; peak {peak_profit*lev:+.1f}% margin &middot; lev {lev}x</div>'
            f'<div class="track"><div class="eline" style="left:66.7%"></div>'
            f'<div class="dot" style="left:{posp:.0f}%;background:{col}"></div></div></div>')
        pcol = st.columns([22, 1])
        pcol[0].markdown(card, unsafe_allow_html=True)
        # ✕ close-this-trade button (you click it — closes the position at market on the exchange)
        if can_trade and pcol[1].button("✕", key=f"close_{sym}", help=f"Close {sym} now at market"):
            try:
                t.close_futures(sym)
                t.cancel_open_orders(sym)
                st.toast(f"Closing {sym} at market — clears on the next refresh.")
                st.rerun()
            except Exception as e:
                st.toast(f"Could not close {sym}: {e}")

    # ---- auto suggestions ----
    st.markdown('<div class="h">🏆 Best coins right now <span class="sub">(auto — from the bot\'s live scan)</span></div>',
                unsafe_allow_html=True)
    rec_html = '<div class="sub">Waiting for the bot to scan… (updates once the bot is running).</div>'
    if RECS.exists():
        try:
            rd = json.load(open(RECS))
            picks = [p for p in rd.get("picks", []) if p.get("symbol") not in state][:8]
            if picks:
                rows = ""
                for p in picks:
                    sg = p.get("signal", 0)
                    sc = "#178a3a" if sg > 0 else "#c0392b"
                    sd = "long" if sg > 0 else "short"
                    fire = " 🔥" if p.get("volume_flag") else ""
                    rows += (f'<div class="fitem"><span style="font-weight:500;min-width:100px">{p.get("symbol")}</span>'
                             f'<span style="color:{sc};min-width:46px">{sd}</span>'
                             f'<span style="color:#667085;flex:1">{p.get("strategy", "")}</span>'
                             f'<span style="min-width:70px">{p.get("confirmations", "?")} agree</span>'
                             f'<span style="color:#667085;min-width:80px">net {p.get("net", "?")}%{fire}</span></div>')
                rec_html = f'<div class="feed">{rows}</div><div class="sub">updated {rd.get("time", "")}</div>'
            else:
                rec_html = '<div class="sub">All top picks are already open, or none pass the rules right now.</div>'
        except Exception:
            pass
    st.markdown(rec_html, unsafe_allow_html=True)

    # ---- activity feed ----
    st.markdown('<div class="h">📡 Live activity</div>', unsafe_allow_html=True)
    items = ""
    if LOG.exists():
        # drop the per-cycle HOLD status spam so real events (opens/closes/news) always show
        _ev = [ln for ln in LOG.read_text(errors="ignore").splitlines() if " HOLD " not in ln]
        for ln in reversed(_ev[-400:]):
            parts = ln.split(" ", 1)
            if len(parts) < 2:
                continue
            tm, msg = parts[0], parts[1]
            low = msg.lower()
            if msg.startswith("OPEN"):
                c, m = "#178a3a", "opened " + msg[5:].split(" | ")[0]
            elif msg.startswith("CLOSE"):
                c = "#c0392b" if "stop" in low else "#178a3a"
                m = "closed " + msg[6:]
            elif "news trade" in low:
                c, m = "#178a3a", msg
            elif "auto bot start" in low:
                c, m = "#178a3a", msg
            elif any(w in low for w in ("skip", "cooldown", "blocked", "batch limit")):
                c, m = "#98a2b3", msg.split(" | ")[0]
            else:
                continue
            items += (f'<div class="fitem"><span class="ft">{tm[:5]}</span>'
                      f'<span style="color:{c}">●</span><span>{m[:90]}</span></div>')
            if items.count("fitem") >= 12:
                break
    feed_body = items or "<div class='sub'>No activity yet.</div>"
    st.markdown(f'<div class="feed">{feed_body}</div>', unsafe_allow_html=True)

    # ---- plain-English diary ----
    st.markdown('<div class="h">📖 What the bot did <span class="sub">(plain English)</span></div>',
                unsafe_allow_html=True)
    if DIARY.exists():
        lines = [ln for ln in DIARY.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        rows = ""
        for ln in reversed(lines[-400:][-25:]):
            when, _, what = ln.partition("  ")
            c = "#178a3a" if "Opened" in what or "PROFIT" in what else (
                "#c0392b" if "LOSS" in what else "#98a2b3")
            rows += (f'<div class="fitem"><span class="ft">{when[11:16]}</span>'
                     f'<span style="color:{c}">●</span><span>{what}</span></div>')
        st.markdown(f'<div class="feed">{rows}</div>', unsafe_allow_html=True)
        st.caption("Full diary: activity.log on the server")
    else:
        st.markdown('<div class="sub">No diary yet - starts filling once the bot runs.</div>',
                    unsafe_allow_html=True)

    # ---- performance moved to the dedicated 📊 Analysis page ----
    st.markdown('<div class="h">📈 Performance <span class="sub">→ open the <b>📊 Analysis</b> page in the '
                'sidebar for daily results, cumulative P/L, and the strategy breakdown.</span></div>',
                unsafe_allow_html=True)

    # ---- trade history (simple, real prices) ----
    if hist is not None and len(hist):
        st.markdown('<div class="h">📒 Trade history <span class="sub">(newest first · '
                    'prices are real, target is an estimate)</span></div>', unsafe_allow_html=True)
        h = hist.copy()
        for c in ("entry", "exit", "price_move_%", "pnl_%"):
            if c in h.columns:
                h[c] = pd.to_numeric(h[c], errors="coerce")
        h["_ct"] = pd.to_datetime(h["close_time"], errors="coerce")
        h = h.sort_values("_ct", ascending=False).head(60)

        # each strategy's typical WINNING price move (used to estimate a target price)
        wins = hist.copy()
        wins["pm"] = pd.to_numeric(wins["price_move_%"], errors="coerce")
        wins = wins[wins["pm"] > 0]
        by_strat = wins.groupby("strategy")["pm"].mean().to_dict() if len(wins) else {}
        overall_win = wins["pm"].mean() if len(wins) else 5.0

        def fmt_when(v):
            t = pd.to_datetime(v, errors="coerce")
            return t.strftime("%b %d, %H:%M") if pd.notna(t) else "—"

        def why_closed(reason, pnl):
            r = str(reason).lower()
            if "trail" in r:
                return "🟢 Trailing stop"
            if "signal" in r:
                return "🔄 Signal flipped"
            if pnl is not None and pnl < 0:
                return "🛑 Hit stop"
            return "🛑 Closed"

        rows = ""
        for _, r in h.iterrows():
            coin = str(r["coin"]).replace("USDT", "")
            islong = str(r["side"]).upper().startswith("L")
            bet = ("🟢 Long", "#178a3a") if islong else ("🔴 Short", "#c0392b")
            entry, ex = r["entry"], r["exit"]
            mv = by_strat.get(r["strategy"], overall_win)
            if pd.isna(mv):
                mv = overall_win
            tgt = entry * (1 + (mv if islong else -mv) / 100) if pd.notna(entry) else None
            pnl = None if pd.isna(r["pnl_%"]) else float(r["pnl_%"])
            pu = r.get("pnl_usdt")                      # REAL $ from the actual trade size (fraction-aware)
            pu = None if pu is None or pd.isna(pu) else float(pu)
            _sz = r.get("size_usdt")                    # margin actually used on this trade
            sz_txt = f"${float(_sz):.0f}" if pd.notna(_sz) else "—"
            pcol = "#178a3a" if (pnl or 0) >= 0 else "#c0392b"
            usd = "—" if pu is None else (f"+${pu:.1f}" if pu >= 0 else f"−${abs(pu):.1f}")
            res = "—" if pnl is None else f"{pnl:+.1f}%"
            rows += (
                f'<tr><td class="rsub">{fmt_when(r["open_time"])}</td>'
                f'<td style="font-weight:500">{coin}</td>'
                f'<td style="color:{bet[1]}">{bet[0]}</td>'
                f'<td>{fnum(entry)}</td>'
                f'<td class="rsub">{fnum(tgt)}</td>'
                f'<td>{fnum(ex)}</td>'
                f'<td class="rsub">{sz_txt}</td>'
                f'<td style="color:{pcol};font-weight:500">{res}'
                f'<div class="rsub" style="color:{pcol}">{usd}</div></td>'
                f'<td class="rsub">{why_closed(r["exit_reason"], pnl)}</td></tr>')
        st.markdown(
            '<div class="hwrap"><table class="htbl"><thead><tr>'
            '<th>When</th><th>Coin</th><th>Bet</th><th>Went in at</th>'
            '<th>Target (est.)</th><th>Got out at</th><th>Size</th><th>Result</th><th>Why it closed</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)
        st.caption('Size = margin used on that trade. Result shows the % return on the margin and, below it, '
                   'the real profit/loss in $ on that exact trade (“—” for older trades saved before size '
                   'tracking). "Target (est.)" is a rough guide only — the bot rides winners with a trailing stop.')

    if err:
        st.caption(f"(live account data unavailable — check .env keys: {err})")
    st.caption("↻ Auto-refreshes every 20 seconds")


if not can_trade and PIN:
    st.info("Enter your trading PIN in the sidebar to control the bot. Dashboard is view-only until then.")
dashboard()
