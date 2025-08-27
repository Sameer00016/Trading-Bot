import streamlit as st
import pandas as pd
import numpy as np
from bot_core import (
    ExchangeManager,
    calculate_indicators,
    generate_signal,
    backtest_strategy,
    send_telegram_message,
)

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="AI Trading Bot", layout="wide")

st.title("🤖 AI Trading Bot with TA + Probability + Calculus")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
exchange_name = st.sidebar.selectbox("Select Exchange", ["binance", "mexc"])
symbol = st.sidebar.text_input("Trading Pair", "BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"])
limit = st.sidebar.slider("Data Points", min_value=50, max_value=500, value=200)
mock_mode = st.sidebar.checkbox("Mock Mode", value=True)

# Initialize exchange manager
exchange = ExchangeManager(exchange_name=exchange_name, mock=mock_mode)

# Fetch Data
st.subheader("📊 Market Data")
try:
    df = exchange.fetch_ohlcv(symbol, timeframe, limit)
    df = calculate_indicators(df)

    st.dataframe(df.tail(10))

    # Generate Signal
    st.subheader("🚦 Trading Signal")
    signal = generate_signal(df)
    st.success(f"Signal: {signal}")

    # Backtest
    st.subheader("📈 Backtest Results")
    try:
        results = backtest_strategy(df)
        st.write(results)
    except Exception as e:
        st.error(f"Backtest failed: {e}")

    # Probability + Calculus Forecast
    st.subheader("📐 Probability & Calculus Forecast")
    try:
        prices = df["close"].values
        returns = np.diff(prices) / prices[:-1]

        prob_up = np.mean(returns > 0)
        prob_down = np.mean(returns < 0)

        # calculus: slope of EMA20 (derivative)
        ema20 = df["EMA20"].values
        slope = np.gradient(ema20)[-1]

        forecast = "UP" if prob_up > prob_down and slope > 0 else "DOWN"

        st.info(f"Probability Market Goes UP: {prob_up:.2%}")
        st.info(f"Probability Market Goes DOWN: {prob_down:.2%}")
        st.success(f"📊 Calculus-adjusted Forecast: {forecast}")
    except Exception as e:
        st.error(f"Forecast failed: {e}")

    # Telegram Alert
    if st.button("Send Signal to Telegram"):
        sent = send_telegram_message(f"Signal: {signal} | Forecast: {forecast}")
        if sent:
            st.success("Telegram alert sent ✅")
        else:
            st.warning("Telegram not configured ⚠️")

except Exception as e:
    st.error(f"Failed to fetch data: {e}")
