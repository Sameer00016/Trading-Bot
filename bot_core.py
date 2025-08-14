# bot_core.py — pure pandas/numpy indicators + BB, multi-exchange, mock-safe

import time
from functools import wraps

import numpy as np
import pandas as pd

# ccxt is optional: we fall back to mock mode if unavailable or init fails
try:
    import ccxt  # type: ignore
    _CCXT_OK = True
except Exception:
    ccxt = None  # type: ignore
    _CCXT_OK = False

import requests


# =========================
# Retry with exponential backoff
# =========================
def backoff_retry(max_retries=5, base_delay=1.0, max_delay=30.0):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt >= max_retries - 1:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
            # Should never reach here
            return fn(*args, **kwargs)
        return wrapped
    return deco


# =========================
# Core technical indicators (pure pandas/numpy)
# =========================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing via EMA with alpha=1/length
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    close_prev = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - close_prev).abs()
    tr3 = (df["low"] - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder's ATR via EMA (alpha=1/length)
    return tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

def bollinger_bands(series: pd.Series, length: int = 20, n_std: float = 2.0):
    mid = series.rolling(length, min_periods=length).mean()
    sd = series.rolling(length, min_periods=length).std(ddof=0)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    # %B: where price sits within the bands
    pb = (series - lower) / (upper - lower)
    return mid, upper, lower, width, pb


# =========================
# Exchange Manager (mock-safe, multi-exchange)
# =========================
class ExchangeManager:
    def __init__(self, exchange_name: str, api_key: str | None = None, api_secret: str | None = None, mock_mode: bool = True):
        """
        If ccxt is unavailable or init fails, we silently fall back to mock_mode.
        """
        self.mock_mode = mock_mode or not _CCXT_OK
        self.exchange = None

        if not self.mock_mode:
            try:
                klass = getattr(ccxt, exchange_name.lower(), None)
                if klass is None:
                    raise ValueError(f"Unsupported exchange: {exchange_name}")
                self.exchange = klass({"apiKey": api_key or "", "secret": api_secret or "", "enableRateLimit": True})
                # Not all exchanges require this, but safe:
                if hasattr(self.exchange, "load_markets"):
                    self.exchange.load_markets()
            except Exception:
                # Fallback to mock if anything goes wrong
                self.exchange = None
                self.mock_mode = True

    @backoff_retry(max_retries=4)
    def get_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 300) -> pd.DataFrame:
        if self.mock_mode or self.exchange is None:
            # Generate a smooth synthetic series with light trend + noise
            idx = pd.date_range(end=pd.Timestamp.utcnow().floor("min"), periods=limit, freq=_tf_to_freq(timeframe))
            base = np.cumsum(np.random.normal(0, 0.3, size=limit)) + 100
            drift = np.linspace(0, np.random.uniform(-3, 3), limit)
            close = base + drift
            high = close + np.abs(np.random.normal(0.4, 0.2, size=limit))
            low = close - np.abs(np.random.normal(0.4, 0.2, size=limit))
            open_ = close + np.random.normal(0, 0.2, size=limit)
            vol = np.abs(np.random.normal(10, 3, size=limit))
            df = pd.DataFrame(
                {"timestamp": idx, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
            )
            return df
        else:
            data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
            return df


def _tf_to_freq(tf: str) -> str:
    """Map ccxt timeframe to pandas frequency for mock series."""
    mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1H", "4h": "4H", "1d": "1D"}
    return mapping.get(tf, "1D")


# =========================
# Trading Strategy
# =========================
class TradingStrategy:
    def __init__(self,
                 ema_fast: int = 20,
                 ema_slow: int = 50,
                 rsi_len: int = 14,
                 atr_len: int = 14,
                 bb_len: int = 20,
                 bb_n: float = 2.0):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_len = rsi_len
        self.atr_len = atr_len
        self.bb_len = bb_len
        self.bb_n = bb_n

    def _ensure_sorted(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.index.is_monotonic_increasing:
            df = df.sort_values("timestamp")
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns original df + indicator columns + 'signal' column in {BUY, SELL, HOLD}.
        """
        df = df.copy()
        df = self._ensure_sorted(df)

        # Indicators
        df["EMA_FAST"] = ema(df["close"], self.ema_fast)
        df["EMA_SLOW"] = ema(df["close"], self.ema_slow)
        df["RSI"] = rsi(df["close"], self.rsi_len)
        macd_line, macd_sig, macd_hist = macd(df["close"])
        df["MACD"] = macd_line
        df["MACD_SIGNAL"] = macd_sig
        df["MACD_HIST"] = macd_hist
        df["ATR"] = atr(df, self.atr_len)

        bb_mid, bb_up, bb_lo, bb_w, bb_pb = bollinger_bands(df["close"], self.bb_len, self.bb_n)
        df["BB_MID"] = bb_mid
        df["BB_UPPER"] = bb_up
        df["BB_LOWER"] = bb_lo
        df["BB_WIDTH"] = bb_w.fillna(0)
        df["BB_PB"] = bb_pb.clip(0, 1).fillna(0.5)

        # Default HOLD
        df["signal"] = "HOLD"

        # Signal logic (conservative & simple)
        # BUY: bullish EMA stack, MACD momentum positive, price above BB mid, RSI supportive
        buy_mask = (
            (df["EMA_FAST"] > df["EMA_SLOW"]) &
            (df["MACD_HIST"] > 0) &
            (df["close"] > df["BB_MID"]) &
            (df["RSI"] > 50)
        )

        # SELL: bearish EMA stack, MACD momentum negative, price below BB mid, RSI weak
        sell_mask = (
            (df["EMA_FAST"] < df["EMA_SLOW"]) &
            (df["MACD_HIST"] < 0) &
            (df["close"] < df["BB_MID"]) &
            (df["RSI"] < 50)
        )

        df.loc[buy_mask, "signal"] = "BUY"
        df.loc[sell_mask, "signal"] = "SELL"

        # Clean early NaNs (first N bars) -> HOLD
        needed_cols = ["EMA_FAST", "EMA_SLOW", "RSI", "MACD_HIST", "BB_MID"]
        early_nan = df[needed_cols].isna().any(axis=1)
        df.loc[early_nan, "signal"] = "HOLD"

        return df


# =========================
# Backtester (very simple count-based example)
# =========================
class Backtester:
    def __init__(self, strategy: TradingStrategy):
        self.strategy = strategy

    def run(self, df: pd.DataFrame) -> dict:
        if df is None or df.empty:
            return {"total_trades": 0, "buy_signals": 0, "sell_signals": 0}

        sigs = self.strategy.generate_signals(df)
        buys = int((sigs["signal"] == "BUY").sum())
        sells = int((sigs["signal"] == "SELL").sum())
        total = buys + sells

        # (Optional) super-light PnL proxy: price change after signal next bar
        pnl = 0.0
        closes = sigs["close"].values
        signals = sigs["signal"].values
        for i in range(len(sigs) - 1):
            if signals[i] == "BUY":
                pnl += closes[i + 1] - closes[i]
            elif signals[i] == "SELL":
                pnl += closes[i] - closes[i + 1]

        return {"total_trades": total, "buy_signals": buys, "sell_signals": sells, "pnl_proxy": float(pnl)}


# =========================
# Telegram notifier
# =========================
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id

    def send_message(self, text: str) -> bool:
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            r = requests.post(url, json={"chat_id": self.chat_id, "text": text}, timeout=10)
            return r.status_code == 200
        except Exception:
            return False
