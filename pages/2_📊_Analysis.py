"""📊 Analysis — daily P/L vs capital, cumulative curves ($ and %), and strategy breakdown.
All dollar figures come from each trade's REAL recorded size (pnl_usdt); the strategy table uses
the margin-return % (pnl_%), which every trade has."""
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
HIST = ROOT / "trade_history.csv"

st.set_page_config(page_title="RL - Trade bot · Analysis", page_icon="📊", layout="wide")
st.markdown("""<style>
.h{font-size:16px;font-weight:500;margin:1.4rem 0 .5rem}
.sub{color:#667085;font-size:12px;font-weight:400}
.hwrap{max-height:520px;overflow:auto;background:#fff;border:1px solid #e6e9ef;border-radius:12px}
.htbl{width:100%;border-collapse:collapse;font-size:13px}
.htbl th{position:sticky;top:0;background:#f5f7fa;text-align:right;color:#667085;font-weight:500;
         font-size:11px;padding:.55rem .75rem;border-bottom:1px solid #e6e9ef;white-space:nowrap}
.htbl th:first-child,.htbl td:first-child{text-align:left}
.htbl td{padding:.5rem .75rem;border-bottom:1px solid #eef1f6;white-space:nowrap;text-align:right;
         font-variant-numeric:tabular-nums}
.htbl tr:last-child td{border-bottom:none}
</style>""", unsafe_allow_html=True)

st.title("📊 Analysis")

# ---------- load history ----------
if not HIST.exists():
    st.info("No trade history yet — it fills as the bot closes trades."); st.stop()
try:
    hist = pd.read_csv(HIST)
except Exception as e:
    st.error(f"Could not read trade history: {e}"); st.stop()
if not len(hist):
    st.info("No trades recorded yet."); st.stop()

hist["pct"] = pd.to_numeric(hist["pnl_%"], errors="coerce")          # margin-return % (every row has it)
hist["usd"] = pd.to_numeric(hist.get("pnl_usdt"), errors="coerce")   # real $ (rows with a recorded size)
hist["ct"] = pd.to_datetime(hist["close_time"], errors="coerce")
hist = hist.dropna(subset=["ct"]).sort_values("ct")

# ---------- capital ----------
balance = None
try:
    from trader.client import BinanceTrader
    balance = BinanceTrader().futures_balance().get("USDT")
except Exception:
    pass

cash = hist.dropna(subset=["usd"]).copy()                # trades with a REAL $ figure
total_pnl = float(cash["usd"].sum())
current_cap = float(balance) if balance else round(5000.0 + total_pnl, 2)
start_cap = max(current_cap - total_pnl, 1.0)            # capital before the recorded trades

st.caption(f"Capital now **${current_cap:,.0f}** · recorded realised P/L **${total_pnl:+,.0f}** · "
           f"inferred starting capital **${start_cap:,.0f}** · {len(cash)} trades with a recorded size. "
           f"(Trades closed before size-tracking aren't in the $ views below.)")

GREEN, RED, BLUE = "#178a3a", "#c0392b", "#4C9AFF"


def zero_rule():
    return alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#c8ced6", strokeDash=[3, 3]).encode(y="y:Q")


# ======================================================= 1) DAILY RESULTS
st.markdown('<div class="h">📅 Daily results <span class="sub">— each 24h in $ and % of capital</span></div>',
            unsafe_allow_html=True)
if len(cash):
    d = cash.copy()
    d["date"] = d["ct"].dt.date
    daily = d.groupby("date")["usd"].sum().reset_index(name="pnl_usd").sort_values("date")
    opens, cap = [], start_cap
    for _, row in daily.iterrows():
        opens.append(cap); cap += row["pnl_usd"]
    daily["opening"] = opens
    daily["closing"] = daily["opening"] + daily["pnl_usd"]
    daily["pnl_pct"] = daily["pnl_usd"] / daily["opening"] * 100
    rows = ""
    for _, r in daily.sort_values("date", ascending=False).iterrows():
        c = GREEN if r["pnl_usd"] >= 0 else RED
        rows += (f'<tr><td>{r["date"]:%b %d, %Y}</td>'
                 f'<td>${r["opening"]:,.0f}</td>'
                 f'<td style="color:{c};font-weight:500">{r["pnl_pct"]:+.2f}%</td>'
                 f'<td style="color:{c};font-weight:500">{r["pnl_usd"]:+,.2f}</td>'
                 f'<td>${r["closing"]:,.0f}</td></tr>')
    st.markdown('<div class="hwrap"><table class="htbl"><thead><tr>'
                '<th>Date</th><th>Opening capital</th><th>P/L %</th><th>P/L $</th><th>Closing capital</th>'
                f'</tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)
else:
    st.info("Daily $ results start once trades close with a recorded size (current bot version).")

# ======================================================= 2) CUMULATIVE % OF CAPITAL
st.markdown('<div class="h">📈 Cumulative P/L — % of capital</div>', unsafe_allow_html=True)
if len(cash):
    c2 = cash.copy()
    c2["cum_pct"] = c2["usd"].cumsum() / start_cap * 100
    line = alt.Chart(c2).mark_line(color=BLUE, strokeWidth=2.5, interpolate="monotone").encode(
        x=alt.X("ct:T", title=None),
        y=alt.Y("cum_pct:Q", title="Cumulative P/L (% of capital)", axis=alt.Axis(format="+.1f")))
    st.altair_chart((zero_rule() + line).properties(height=240), use_container_width=True)
    st.caption("Real, size-aware return as a percentage of your capital — not a sum of per-trade percentages.")
else:
    st.info("Fills as trades close with a recorded size.")

# ======================================================= 3) CUMULATIVE $
st.markdown('<div class="h">💵 Cumulative P/L — $</div>', unsafe_allow_html=True)
if len(cash):
    c3 = cash.copy()
    c3["cum_usd"] = c3["usd"].cumsum()
    line = alt.Chart(c3).mark_line(color=GREEN, strokeWidth=2.5, interpolate="monotone").encode(
        x=alt.X("ct:T", title=None),
        y=alt.Y("cum_usd:Q", title="Cumulative P/L ($)", axis=alt.Axis(format="$,.0f")))
    st.altair_chart((zero_rule() + line).properties(height=240), use_container_width=True)
    st.caption(f"Actual dollars gained/lost, cumulative — on a starting capital of ${start_cap:,.0f}.")
else:
    st.info("Fills as trades close with a recorded size.")

# ======================================================= 4) STRATEGY ANALYSIS
st.markdown('<div class="h">🧠 Strategy analysis <span class="sub">— normal strategy trades, most-used first</span></div>',
            unsafe_allow_html=True)
strat = hist[hist["type"] == "strategy"].copy() if "type" in hist.columns else hist.copy()
strat = strat.dropna(subset=["pct"])
if len(strat):
    g = strat.groupby("strategy").agg(
        trades=("pct", "size"), avg=("pct", "mean"),
        win=("pct", lambda x: (x > 0).mean() * 100),
        usd=("usd", "sum")).reset_index().sort_values("trades", ascending=False)

    def remark(n, avg, win):
        if n < 4:
            return f"small sample ({n} trades)"
        if avg > 1.5 and win >= 50:
            return f"{win:.0f}% win · strong — keep"
        if avg > 0:
            return f"{win:.0f}% win · net positive — keep"
        return f"{win:.0f}% win · losing — prune candidate"

    rows = ""
    for _, r in g.iterrows():
        c = GREEN if r["avg"] >= 0 else RED
        usd = f' · ${r["usd"]:+,.0f}' if pd.notna(r["usd"]) and r["usd"] != 0 else ""
        rows += (f'<tr><td>{r["strategy"]}</td>'
                 f'<td>{int(r["trades"])}</td>'
                 f'<td style="color:{c};font-weight:500">{r["avg"]:+.2f}%</td>'
                 f'<td style="text-align:left;color:#667085">{remark(int(r["trades"]), r["avg"], r["win"])}{usd}</td></tr>')
    st.markdown('<div class="hwrap"><table class="htbl"><thead><tr>'
                '<th>Strategy</th><th>Trades</th><th>Profit % (avg/trade)</th>'
                '<th style="text-align:left">Remarks</th>'
                f'</tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)
    st.caption("Profit % = average return on the margin per trade. Remarks = win rate + a keep/prune read, "
               "and total $ where a size was recorded.")
else:
    st.info("No strategy trades recorded yet.")

st.caption("↻ Reopen this page to refresh the numbers.")
