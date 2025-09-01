import os
from typing import Dict, Any

import ccxt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


# ========= Helpers: indicators (pure pandas/numpy) =========
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    up_ema = pd.Series(up, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    down_ema = pd.Series(down, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    rs = up_ema / (down_ema + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return pd.DataFrame(
        {
            "MACD": line,
            "MACD_SIGNAL": sig,
            "MACD_HIST": hist,
            "EMA_FAST": ema_fast,
            "EMA_SLOW": ema_slow,
        },
        index=series.index,
    )


def bollinger(series: pd.Series, length: int = 20, std_mult: float = 2.0) -> pd.DataFrame:
    mid = series.rolling(length, min_periods=length).mean()
    std = series.rolling(length, min_periods=length).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower).replace(0.0, np.nan)
    # %B scaled 0..1 (can go outside in extremes)
    pb = (series - lower) / width
    return pd.DataFrame(
        {"BB_MID": mid, "BB_UPPER": upper, "BB_LOWER": lower, "BB_PB": pb},
        index=series.index,
    )


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Work on a copy; ensure unique columns first
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Core indicators
    macd_df = macd(df["close"])
    bb_df = bollinger(df["close"])
    rsi_series = rsi(df["close"], period=14)
    slope = np.gradient(df["close"].to_numpy())
    slope_series = pd.Series(slope, index=df.index, name="PRICE_SLOPE")

    # Drop existing indicator cols if present (avoid duplicates)
    drop_cols = set(macd_df.columns) | set(bb_df.columns) | {"RSI", "PRICE_SLOPE"}
    existing = drop_cols.intersection(df.columns)
    if existing:
        df = df.drop(columns=list(existing))

    # Join all
    df = df.join([macd_df, bb_df])
    df["RSI"] = rsi_series
    df["PRICE_SLOPE"] = slope_series

    # Clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Final de-dup (defensive)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


# ========= Exchange manager =========
class ExchangeManager:
    def __init__(self, exchange_name: str = "binance", mock: bool = True):
        self.exchange_name = exchange_name
        self.mock = mock
        self.exchange = None

        if not mock:
            api_key = os.getenv(f"{exchange_name.upper()}_API_KEY")
            api_secret = os.getenv(f"{exchange_name.upper()}_API_SECRET")
            if not api_key or not api_secret:
                raise ValueError(
                    f"Missing API keys for {exchange_name}. "
                    f"Set {exchange_name.upper()}_API_KEY and {exchange_name.upper()}_API_SECRET in .env"
                )

            exchange_class = getattr(ccxt, exchange_name)
            self.exchange = exchange_class(
                {"apiKey": api_key, "secret": api_secret, "enableRateLimit": True}
            )

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 300) -> pd.DataFrame:
        """
        Returns a dataframe with columns:
        timestamp (datetime), open, high, low, close, volume + indicators
        """
        if self.mock:
            # synthetic but realistic-ish data
            freq = {"m": "min", "h": "H", "d": "D"}
            unit = "H"
            if timeframe.endswith("m"):
                unit = "min"
            elif timeframe.endswith("h"):
                unit = "H"
            elif timeframe.endswith("d"):
                unit = "D"

            dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq=unit)
            base = np.cumsum(np.random.randn(limit)) + 20000.0
            close = base + np.random.randn(limit) * 20.0
            high = close + np.abs(np.random.randn(limit) * 15.0)
            low = close - np.abs(np.random.randn(limit) * 15.0)
            open_ = close + np.random.randn(limit) * 10.0
            vol = np.abs(np.random.randn(limit) * 50.0) + 10.0

            df = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": vol,
                }
            )
        else:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(
                raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Ensure canonical dtypes
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Add indicators and return
        return add_indicators(df)


# ========= Strategy & backtest =========
def generate_signal(df: pd.DataFrame) -> str:
    """
    Simple decision on the latest row combining EMA crossover, RSI, MACD histogram, and slope.
    """
    last = df.iloc[-1]
    cond_buy = (
        (last["EMA_FAST"] > last["EMA_SLOW"])
        and (last["RSI"] < 70)
        and (last["MACD_HIST"] > 0)
        and (last["PRICE_SLOPE"] > 0)
    )
    cond_sell = (
        (last["EMA_FAST"] < last["EMA_SLOW"])
        and (last["RSI"] > 30)
        and (last["MACD_HIST"] < 0)
        and (last["PRICE_SLOPE"] < 0)
    )

    if cond_buy and not cond_sell:
        return "BUY"
    if cond_sell and not cond_buy:
        return "SELL"
    return "HOLD"


def backtest_strategy(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Vectorized backtest for the same rule over the whole dataframe.
    """
    # Conditions per row
    buy = (df["EMA_FAST"] > df["EMA_SLOW"]) & (df["RSI"] < 70) & (df["MACD_HIST"] > 0) & (df["PRICE_SLOPE"] > 0)
    sell = (df["EMA_FAST"] < df["EMA_SLOW"]) & (df["RSI"] > 30) & (df["MACD_HIST"] < 0) & (df["PRICE_SLOPE"] < 0)

    signals = np.where(buy & ~sell, "BUY", np.where(sell & ~buy, "SELL", "HOLD"))
    df_bt = df.copy()
    df_bt["Signal"] = signals

    ret = df_bt["close"].pct_change().fillna(0.0)
    pos = pd.Series(0, index=df_bt.index)
    pos[df_bt["Signal"] == "BUY"] = 1
    pos[df_bt["Signal"] == "SELL"] = -1
    pos = pos.shift(1).fillna(0)  # enter on next bar
    equity = (pos * ret).cumsum()

    trades = (df_bt["Signal"] != "HOLD").sum()
    wins = ((pos * ret) > 0).sum()

    result = {
        "Final Return %": round(float(equity.iloc[-1]) * 100, 2),
        "Win Rate %": round((wins / max(1, len(df_bt))) * 100, 2),
        "Trades": int(trades),
    }
    return result


# ========= Telegram =========
def send_telegram_message(message: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False
