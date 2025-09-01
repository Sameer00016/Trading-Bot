import streamlit as st
import numpy as np
from bot_core import ExchangeManager, generate_signal, backtest_strategy, send_telegram_message

st.set_page_config(page_title="AI Trading Bot", layout="wide")
st.title("🤖 AI Trading Bot — Live Signals (No pandas_ta)")

# Sidebar
st.sidebar.header("⚙️ Settings")
exchange_name = st.sidebar.selectbox("Select Exchange", ["binance", "mexc"])
symbol = st.sidebar.text_input("Trading Pair", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
limit = st.sidebar.slider("Data Points", min_value=60, max_value=1000, value=300, step=20)
mock_mode = st.sidebar.checkbox("Mock Mode (no API needed)", value=True)

ex = ExchangeManager(exchange_name=exchange_name, mock=mock_mode)

st.subheader("📊 Market Data")
try:
    df = ex.fetch_ohlcv(symbol, timeframe, limit)
    st.dataframe(df.tail(12))

    # Signal
    st.subheader("🚦 Trading Signal")
    signal = generate_signal(df)
    st.success(f"Signal: {signal}")

    # Backtest
    st.subheader("📈 Backtest Results")
    results = backtest_strategy(df)
    st.json(results)

    # Probability & Calculus Forecast
    st.subheader("📐 Probability & Calculus Forecast")
    prices = df["close"].to_numpy()
    returns = np.diff(prices) / prices[:-1]
    prob_up = float((returns > 0).mean()) if len(returns) else 0.5
    prob_down = 1.0 - prob_up
    slope = float(np.gradient(df["EMA_FAST"].to_numpy())[-1])
    forecast = "UP" if (prob_up > prob_down and slope > 0) else "DOWN"
    st.info(f"Probability UP: {prob_up:.2%}")
    st.info(f"Probability DOWN: {prob_down:.2%}")
    st.success(f"📊 Forecast: {forecast}")

    # Telegram Alerts
    if st.button("📩 Send Signal to Telegram"):
        sent = send_telegram_message(f"{symbol} | Signal: {signal} | Forecast: {forecast}")
        st.success("Telegram alert sent ✅" if sent else "Telegram not configured ⚠️")

except Exception as e:
    st.error(f"Failed to get signals: {e}")
