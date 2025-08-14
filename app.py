import streamlit as st
import os
from bot_core import ExchangeManager, TradingStrategy, Backtester, TelegramNotifier

# Sidebar settings
st.sidebar.title("Trading Bot Settings")

# Select exchange
exchange_name = st.sidebar.selectbox("Select Exchange", ["Binance", "MEXC"])
mock_mode = st.sidebar.checkbox("Mock Mode", value=True)

# API Keys
api_key = st.sidebar.text_input("API Key", type="password")
api_secret = st.sidebar.text_input("API Secret", type="password")

# Trading pair & timeframe
symbol = st.sidebar.text_input("Symbol", value="BTC/USDT")
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"])
limit = st.sidebar.slider("OHLCV Limit", min_value=20, max_value=500, value=100)

# Telegram settings
telegram_token = st.sidebar.text_input("Telegram Bot Token", type="password")
telegram_chat_id = st.sidebar.text_input("Telegram Chat ID")

# Main app
st.title("AI Trading Bot")

if st.button("Run Bot"):
    try:
        # Initialize components
        exchange = ExchangeManager(exchange_name, api_key, api_secret, mock_mode=mock_mode)
        strategy = TradingStrategy()
        backtester = Backtester(strategy)
        notifier = TelegramNotifier(telegram_token, telegram_chat_id)

        # Fetch data
        df = exchange.get_ohlcv(symbol, timeframe=timeframe, limit=limit)

        # Generate signals
        df = strategy.generate_signals(df)

        # Show results
        st.subheader("Signal Data")
        st.dataframe(df.tail(20))

        # Backtest results
        results = backtester.run(df)
        st.subheader("Backtest Summary")
        st.json(results)

        # Send notification
        if telegram_token and telegram_chat_id:
            notifier.send_message(f"Bot run completed. Results: {results}")

    except Exception as e:
        st.error(f"Error running bot: {e}")

