import os
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any
from dotenv import load_dotenv
import requests

load_dotenv()

# ------------------------------
# Exchange Manager
# ------------------------------
class ExchangeManager:
    def __init__(self, exchange_name: str = "binance", mock: bool = True):
        self.exchange_name = exchange_name
        self.mock = mock

        if not mock:
            api_key = os.getenv(f"{exchange_name.upper()}_API_KEY")
            api_secret = os.getenv(f"{exchange_name.upper()}_API_SECRET")
            if not api_key or not api_secret:
                raise ValueError(f"Missing API keys for {exchange_name}. Add them in .env file.")

            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            })
        else:
            self.exchange = None

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
        if self.mock:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq="H")
            prices = np.linspace(20000, 30000, limit) + np.random.randn(limit) * 1000
            df = pd.DataFrame({
                "timestamp": dates,
                "open": prices,
                "high": prices + np.random.rand(limit) * 500,
                "low": prices - np.random.rand(limit) * 500,
                "close": prices,
                "volume": np.random.rand(limit) * 100,
            })
        else:
            data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        return calculate_indicators(df)

# ------------------------------
# Indicators
# ------------------------------
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # EMA
    df["EMA_FAST"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA_SLOW"] = df["close"].ewm(span=26, adjust=False).mean()

    # RSI
    if "RSI" not in df.columns:
        df["RSI"] = ta.rsi(df["close"], length=14)

    # MACD
    macd = ta.macd(df["close"])
    df["MACD"] = macd.iloc[:, 0]
    df["MACD_SIGNAL"] = macd.iloc[:, 1]
    df["MACD_HIST"] = macd.iloc[:, 2]

    # Bollinger Bands
    bbands = ta.bbands(df["close"], length=20, std=2)
    df["BB_MID"] = bbands.iloc[:, 0]
    df["BB_UPPER"] = bbands.iloc[:, 1]
    df["BB_LOWER"] = bbands.iloc[:, 2]
    df["BB_PB"] = bbands.iloc[:, 3]

    # Calculus slope
    df["PRICE_SLOPE"] = np.gradient(df["close"])

    # Final cleanup
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df

# ------------------------------
# Trading Strategy
# ------------------------------
def generate_signal(df: pd.DataFrame) -> str:
    last = df.iloc[-1]

    if last["EMA_FAST"] > last["EMA_SLOW"] and last["RSI"] < 70 and last["MACD_HIST"] > 0 and last["PRICE_SLOPE"] > 0:
        return "BUY"
    elif last["EMA_FAST"] < last["EMA_SLOW"] and last["RSI"] > 30 and last["MACD_HIST"] < 0 and last["PRICE_SLOPE"] < 0:
        return "SELL"
    else:
        return "HOLD"

# ------------------------------
# Backtesting
# ------------------------------
def backtest_strategy(df: pd.DataFrame) -> Dict[str, Any]:
    df["Signal"] = df.apply(lambda _: generate_signal(df), axis=1)
    df["Return"] = df["close"].pct_change().fillna(0)
    df["Strategy"] = df["Signal"].shift(1).map({"BUY": 1, "SELL": -1, "HOLD": 0}).fillna(0)
    df["Equity"] = (df["Strategy"] * df["Return"]).cumsum()

    return {
        "Final Return %": round(df["Equity"].iloc[-1] * 100, 2),
        "Win Rate %": round((df["Strategy"] > 0).mean() * 100, 2),
        "Trades": int((df["Strategy"] != 0).sum())
    }

# ------------------------------
# Telegram Notifier
# ------------------------------
def send_telegram_message(message: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload)
        return True
    except Exception:
        return False
