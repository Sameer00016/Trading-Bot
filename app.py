import os, time
import streamlit as st
import pandas as pd
from datetime import datetime
from bot_core import (
    SUPPORTED_EXCHANGES, ExchangeManager, generate_signal, suggest_pairs,
    format_signal_for_display, bitnodes_snapshot, send_telegram, fetch_htf_alignment,
    run_backtest
)

# Secrets helper
def _secret(key, default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

LOGIN_CODE = _secret("LOGIN_CODE", "Sam0316")
TELE_TOKEN = _secret("TELEGRAM_TOKEN", "")
TELE_CHAT  = _secret("TELEGRAM_CHAT_ID", "")

st.set_page_config(page_title="Trading Bot", layout="wide")
st.markdown("<h2 style='text-align:center'>🤖 Trading Bot — Live Signals (Upgraded)</h2>", unsafe_allow_html=True)

# Auth
pwd = st.text_input("Enter access code", type="password")
if pwd != LOGIN_CODE:
    st.warning("Access denied. Enter the correct access code.")
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
exchange_name = st.sidebar.selectbox("Exchange", SUPPORTED_EXCHANGES, index=0)
timeframe = st.sidebar.selectbox("Timeframe", ["1m","5m","15m","1h","4h"], index=1)
auto = st.sidebar.checkbox("Auto-refresh")
interval = st.sidebar.number_input("Refresh seconds (auto)", min_value=5, max_value=120, value=20, step=5)

# Risk settings
st.sidebar.subheader("Risk")
equity = st.sidebar.number_input("Account Equity", min_value=100.0, value=1000.0, step=100.0)
risk_pct = st.sidebar.slider("Risk % per Trade", min_value=0.1, max_value=3.0, value=1.0, step=0.1)

ex = ExchangeManager(exchange_name)

# Pair select with suggestions
with st.expander("🔎 Suggest pairs (volume, volatility, breakout)"):
    if st.button("Scan & Suggest"):
        with st.spinner("Scanning pairs..."):
            suggestions = suggest_pairs(ex, max_pairs=10, timeframe="5m")
        if suggestions:
            st.success("Suggestions ready. Click to copy:")
            for s in suggestions:
                st.code(s)
        else:
            st.info("No suggestions available (mock mode or rate-limited).")

all_syms = ex.symbols_usdt(400)
pair = st.selectbox("Select pair", options=all_syms, index=0)

cols = st.columns(4)
send_to_telegram = cols[0].checkbox("Send Telegram", value=True)
run = cols[1].button("Get Signal Now")
run_backtest_btn = cols[2].button("Run Backtest (quick)")
st.caption("Indicators: RSI, MACD, EMA(21), ATR(14), Bollinger width + orderbook imbalance → ML probability + filters. Use for research only; not financial advice.")

if auto:
    st.experimental_set_query_params(_=str(time.time()))

# Main action
if run or auto:
    with st.spinner("Fetching data & analyzing..."):
        try:
            df = ex.fetch_ohlcv(pair, timeframe=timeframe)
            ob = None
            try:
                ob = ex.fetch_orderbook(pair, limit=25)
            except Exception:
                ob = None

            ob_imb = 0.5 if ob is None else (lambda ob_: (sum(v for _,v in ob_["bids"][:20]) /
                                                          max(1e-9, sum(v for _,v in ob_["bids"][:20]) + sum(v for _,v in ob_["asks"][:20]))))(ob)

            # higher timeframe confirmation
            htf_align = fetch_htf_alignment(ex, pair, timeframe)

            sig = generate_signal(df, ob_imb, htf_align=htf_align, equity=equity, risk_pct=risk_pct)
            sig["pair"] = pair
            sig["exchange"] = exchange_name

            st.subheader("Signal")
            st.markdown(format_signal_for_display(sig))

            # Chart
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Candlestick(x=df["datetime"], open=df["open"], high=df["high"], low=df["low"], close=df["close"] )])
            fig.update_layout(height=380, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Telegram
            if send_to_telegram:
                ok = send_telegram(sig, TELE_TOKEN, TELE_CHAT, df)
                st.info("Telegram sent ✅" if ok else "Telegram failed ❌ (check token/chat or kaleido)")

            # Optional BTC network context
            if pair.startswith("BTC/"):
                snap = bitnodes_snapshot()
                if snap:
                    st.caption(f"Bitnodes: {snap['nodes']} nodes (as of {datetime.utcfromtimestamp(snap['timestamp']).isoformat()}Z)")

        except Exception as e:
            st.error(f"Error: {e}")

# Backtest section
with st.expander("📈 Quick Backtest"):
    lb = st.number_input("Lookback candles", min_value=400, max_value=3000, value=1200, step=100)
    if run_backtest_btn:
        with st.spinner("Running backtest..."):
            try:
                res = run_backtest(ex, pair, timeframe=timeframe, lookback=int(lb), equity=equity, risk_pct=risk_pct)
                st.json(res)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
