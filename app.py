import streamlit as st
from bot_core import (
    ExchangeManager,
    TradingStrategy,
    Backtester,
    TelegramNotifier
)
import pandas as pd
import os

# Streamlit page setup
st.set_page_config(page_title="AI Trading Bot", layout="wide")
st.title("🤖 AI Trading Bot Dashboard")

# Sidebar settings
st.sidebar.header("Bot Settings")

# Select exchange
exchange_name = st.sidebar.selectbox("Select Exchange", ["Binance", "MEXC"])

# API keys from environment
api_key = os.getenv(f"{exchange_name.upper()}_API_KEY")
api_secret = os.getenv(f"{exchange_name.upper()}_API_SECRET")

# Mock mode toggle
mock_mode = st.sidebar.checkbox("Enable Mock Mode", value=True)

# Trading pair
pair = st.sidebar.text_input("Trading Pair", value="BTC/USDT")

# Timeframe
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=5)

# Initialize exchange manager
exchange_manager = ExchangeManager(exchange_name, api_key, api_secret, mock_mode=mock_mode)

# Load historical data
st.subheader("Market Data")
try:
    df = exchange_manager.get_ohlcv(pair, timeframe)
    if not df.empty:
        st.line_chart(df.set_index("timestamp")["close"])
    else:
        st.warning("No market data found.")
except Exception as e:
    st.error(f"Error loading market data: {e}")
    df = pd.DataFrame()

# Strategy execution
if st.sidebar.button("Run Strategy") and not df.empty:
    strategy = TradingStrategy()
    signals = strategy.generate_signals(df)
    st.subheader("Trading Signals")
    st.dataframe(signals.tail(20))

# Backtest section
st.sidebar.header("Backtesting")
if st.sidebar.button("Run Backtest") and not df.empty:
    backtester = Backtester(TradingStrategy())
    results = backtester.run(df)
    st.subheader("Backtest Results")
    st.write(results)

# Telegram notifications (optional)
if st.sidebar.button("Send Test Alert"):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        TelegramNotifier(token, chat_id).send_message("🚀 Test alert from AI Trading Bot")
        st.success("Test alert sent.")
    else:
        st.warning("Telegram token or chat ID not set in .env")
