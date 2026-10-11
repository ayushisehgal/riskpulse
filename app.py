"""Live demo dashboard.   streamlit run app.py"""
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from riskpulse.ingestion import fetch_all
from riskpulse.pipeline import build

st.set_page_config(page_title="RiskPulse", page_icon="📡", layout="wide")
st.title("📡 RiskPulse — AI/NLP Financial Risk Engine")
st.caption("News + social text → sentiment, event type, impact → live index rebalancing & portfolio stress tests")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Controls")
    live = st.toggle("Use live news (GDELT)", value=False, help="Falls back to bundled sample if offline")
    speed = st.slider("Replay delay (sec / signal)", 0.0, 1.0, 0.08, 0.02)
    thr = st.slider("Stress-test impact threshold", 5.0, 9.0, 7.0, 0.5)
    tilt = st.slider("Rebalancer tilt strength", 0.5, 6.0, 3.0, 0.5)
    go_btn = st.button("▶ Run pipeline", type="primary", use_container_width=True)

if "state" not in st.session_state or go_btn:
    engine, a, b = build(out_path="output/signals.jsonl", a=dict(tilt=tilt), b=dict(threshold=thr))
    docs, warns = fetch_all(live=live)
    for w in warns:
        st.warning(w)
    bar, ph = st.progress(0.0, "Streaming documents through the NLP engine…"), st.empty()
    for i, d in enumerate(docs, 1):
        s = engine.process(d)
        bar.progress(i / len(docs))
        if s and speed:
            ph.caption(f"[{s.source}] {s.text[:90]} → {s.event_type} | sent {s.sentiment_score:+.2f} | impact {s.impact_score}")
            time.sleep(speed)
    bar.empty(); ph.empty()
    st.session_state.state = (engine, a, b)

engine, a, b = st.session_state.state
sig_df = pd.DataFrame([s.to_dict() for s in engine.signals])
sig_df["tickers"] = sig_df.tickers.apply(lambda x: ", ".join(x) or "MARKET")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Signals", len(sig_df))
c2.metric("Avg sentiment", f"{sig_df.sentiment_score.mean():+.2f}")
c3.metric("Max impact", f"{sig_df.impact_score.max():.1f}")
c4.metric("Stress tests fired", len(b.results))

t1, t2, t3, t4 = st.tabs(["🧠 Risk signals", "⚖️ Module A: Rebalancer", "🧪 Module B: Stress test", "🔬 Try your own text"])

# ---------------------------------------------------------------- signals
with t1:
    l, r = st.columns([3, 2])
    with l:
        st.dataframe(sig_df[["timestamp", "source", "tickers", "sentiment_score", "event_type", "impact_score", "text"]]
                     .sort_values("timestamp", ascending=False), use_container_width=True, height=420,
                     column_config={"sentiment_score": st.column_config.ProgressColumn("sentiment", min_value=-1, max_value=1),
                                    "impact_score": st.column_config.ProgressColumn("impact", min_value=0, max_value=10)})
    with r:
        st.plotly_chart(px.scatter(sig_df, x="sentiment_score", y="impact_score", color="event_type", symbol="source",
                                   hover_name="text", title="Sentiment vs impact"), use_container_width=True)
    st.download_button("⬇ Download signals.jsonl", "\n".join(__import__("json").dumps(s.to_dict()) for s in engine.signals),
                       "signals.jsonl")

# ---------------------------------------------------------------- module A
with t2:
    h = a.history_df()
    h = h[h.timestamp != "init"]
    h["step"] = h.groupby("ticker").cumcount() + 1
    st.subheader("Index weights over time (each step = one sentiment-moving signal)")
    st.plotly_chart(px.area(h, x="step", y="weight", color="ticker", groupnorm=None).update_yaxes(tickformat=".0%"),
                    use_container_width=True)
    cur = a.current()
    cl, cr = st.columns(2)
    fig = go.Figure(go.Bar(x=cur.ticker, y=cur.change_bps, marker_color=["#2e7d32" if v >= 0 else "#c62828" for v in cur.change_bps]))
    cl.plotly_chart(fig.update_layout(title="Weight change vs equal-weight (bps)"), use_container_width=True)
    cr.dataframe(cur.style.format({"base": "{:.2%}", "weight": "{:.2%}", "ema_sentiment": "{:+.3f}", "change_bps": "{:+.0f}"}),
                 use_container_width=True, height=380)

# ---------------------------------------------------------------- module B
with t3:
    pf = b.portfolio
    st.write(f"Synthetic wholesale-banking portfolio: **{len(pf)} positions**, **${pf.market_value.sum():,.0f}m** market value")
    if not b.results:
        st.info("No event exceeded the impact threshold. Lower it in the sidebar.")
    else:
        labels = [f"{i+1}. {r.event_type} (impact {r.impact_score}) — {r.trigger_text[:60]}" for i, r in enumerate(b.results)]
        r = b.results[labels.index(st.selectbox("Triggered stress test", labels))]
        m1, m2, m3 = st.columns(3)
        m1.metric("Before", f"${r.before:,.0f}m")
        m2.metric("After", f"${r.after:,.0f}m", f"{r.pnl:,.1f}m ({r.pct:+.2f}%)", delta_color="inverse" if r.pnl < 0 else "normal")
        m3.metric("Severity multiplier", f"{r.severity:.2f}×")
        agg = r.by_asset_type.melt(id_vars="asset_type", value_vars=["market_value", "stressed_value"],
                                   var_name="state", value_name="value_usd_m")
        agg["state"] = agg.state.map({"market_value": "Before", "stressed_value": "After stress"})
        l, rr = st.columns(2)
        l.plotly_chart(px.bar(agg, x="asset_type", y="value_usd_m", color="state", barmode="group",
                              color_discrete_map={"Before": "#1565c0", "After stress": "#c62828"},
                              title="Value before vs after, by asset type"), use_container_width=True)
        worst = r.positions.nsmallest(8, "pnl")[["asset_id", "product", "sector", "rating", "market_value", "pnl"]]
        rr.plotly_chart(px.bar(worst, x="pnl", y="asset_id", orientation="h", color="product",
                               title="Top 8 loss contributors ($m)"), use_container_width=True)
        wf = go.Figure(go.Waterfall(x=["Before"] + list(r.by_asset_type.asset_type) + ["After"],
                                    measure=["absolute"] + ["relative"] * len(r.by_asset_type) + ["total"],
                                    y=[r.before] + list(r.by_asset_type.pnl) + [0]))
        st.plotly_chart(wf.update_layout(title="Portfolio value bridge ($m)"), use_container_width=True)

# ------------------------------------------------------------- try your own
with t4:
    txt = st.text_area("Paste a headline or tweet", "Escalating conflict closes key shipping strait; oil spikes and global stocks tumble")
    src = st.radio("Source type", ["news", "social"], horizontal=True)
    if st.button("Analyze"):
        s = engine.analyze_text(txt + " ", src)
        st.json(s.to_dict())
        st.caption("This signal was also pushed to Module A / Module B (re-run to refresh the tabs).")
