import os, time, math
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# TA
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

# Exchange connector
try:
    import ccxt
    CCXT = True
except Exception:
    CCXT = False

DEFAULT_LIMIT = 300  # more candles for better stats
SAFE_TIMEOUT = 15000

SUPPORTED_EXCHANGES = ["binance", "mexc", "mock"]

def _utcnow():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

class ExchangeManager:
    def __init__(self, name: str):
        self.name = name
        self.mock = (name == "mock" or not CCXT)
        self.ex = None
        if not self.mock:
            try:
                klass = getattr(ccxt, name)
                self.ex = klass({"enableRateLimit": True, "timeout": SAFE_TIMEOUT})
                self.ex.load_markets()
            except Exception as e:
                print(f"[ExchangeManager] falling back to mock ({e})")
                self.mock = True

    def symbols_usdt(self, max_items=200):
        """Return a filtered list of /USDT or major FX symbols."""
        if self.mock or not self.ex:
            return ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","EUR/USD","GBP/USD","USD/JPY"]
        syms = [s for s in self.ex.symbols if ("/USDT" in s) or s in ("EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","NZD/USD","USD/CAD")]
        return syms[:max_items]

    def fetch_ohlcv(self, symbol, timeframe="1m", limit=DEFAULT_LIMIT):
        if self.mock or not self.ex:
            # synthetic series for demo/fallback
            now = int(time.time()*1000)
            prices = np.cumsum(np.random.randn(limit)) + 100
            highs = prices + np.random.rand(limit)*1.5
            lows = prices - np.random.rand(limit)*1.5
            opens = prices + np.random.randn(limit)*0.3
            vols = (np.random.rand(limit)+0.2)*10
            rows = []
            for i in range(limit):
                ts = now - (limit-1-i)*60_000
                rows.append([ts, opens[i], highs[i], lows[i], prices[i], vols[i]])
            df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        # real ccxt
        data = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def fetch_ticker(self, symbol):
        if self.mock or not self.ex:
            return {"last": float(np.random.uniform(50, 70000))}
        t = self.ex.fetch_ticker(symbol)
        return {"last": t.get("last") or t.get("close")}

    def fetch_orderbook(self, symbol, limit=50):
        if self.mock or not self.ex:
            # simple symmetric book
            px = 100 + np.random.randn()*2
            bids = [[px - i*0.1, 1+np.random.rand()] for i in range(limit)]
            asks = [[px + i*0.1, 1+np.random.rand()] for i in range(limit)]
            return {"bids": bids, "asks": asks}
        return self.ex.fetch_order_book(symbol, limit=limit)

    def fetch_top_tickers(self):
        """Return dict of tickers; used for suggesting pairs."""
        if self.mock or not self.ex:
            return {}
        try:
            return self.ex.fetch_tickers()
        except Exception:
            return {}

def analyze(df: pd.DataFrame):
    """Compute indicators and return a copy with fields: rsi, macd, ema, atr."""
    df = df.copy()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"])
    df["macd"] = macd.macd_diff()
    df["ema"]  = EMAIndicator(df["close"], window=21).ema_indicator()
    df["atr"]  = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    return df

def orderbook_imbalance(ob):
    """Return bid/ask volume imbalance in [0,1], >0.5 means bid-dominant."""
    if not ob or "bids" not in ob or "asks" not in ob:
        return 0.5
    bid_vol = sum([v for _, v in ob["bids"][:20]]) or 1e-9
    ask_vol = sum([v for _, v in ob["asks"][:20]]) or 1e-9
    return float(bid_vol / (bid_vol + ask_vol))

def probability_model(df: pd.DataFrame, ob_imb: float):
    """
    Simple probability mix: momentum (ROC), trend (price vs EMA), OB imbalance.
    Returns p_up in [0,1].
    """
    if len(df) < 3:
        return 0.5
    roc = (df["close"].iloc[-1] - df["close"].iloc[-3]) / max(1e-9, abs(df["close"].iloc[-3]))
    trend = 1.0 if df["close"].iloc[-1] > df["ema"].iloc[-1] else 0.0
    # normalize roc into 0..1 via logistic-ish squash
    roc01 = 1/(1+math.exp(-5*roc))
    p = 0.4*roc01 + 0.3*trend + 0.3*ob_imb
    return float(min(max(p, 0.0), 1.0))

def make_trade_plan(df: pd.DataFrame, side: str):
    price = float(df["close"].iloc[-1])
    atr = float(df["atr"].iloc[-1]) if not np.isnan(df["atr"].iloc[-1]) else max(0.5, price*0.002)
    if side == "BUY":
        sl = price - 1.5*atr
        tp = price + 3.0*atr
    elif side == "SELL":
        sl = price + 1.5*atr
        tp = price - 3.0*atr
    else:
        return {"price": price, "stop_loss": None, "take_profit": None, "close_by": None}
    close_by = (datetime.utcnow() + timedelta(minutes=30)).isoformat(timespec="seconds") + "Z"
    return {"price": price, "stop_loss": sl, "take_profit": tp, "close_by": close_by}

def generate_signal(df: pd.DataFrame, ob_imb: float):
    df = analyze(df)
    rsi   = float(df["rsi"].iloc[-1])
    macd  = float(df["macd"].iloc[-1])
    ema   = float(df["ema"].iloc[-1])
    atr   = float(df["atr"].iloc[-1])
    price = float(df["close"].iloc[-1])

    p_up  = probability_model(df, ob_imb)
    # heuristic gating
    long_ok  = (rsi < 40 and macd > 0 and price > ema and p_up > 0.55)
    short_ok = (rsi > 60 and macd < 0 and price < ema and p_up < 0.45)

    if long_ok and not short_ok:
        side = "BUY"
    elif short_ok and not long_ok:
        side = "SELL"
    else:
        side = "WAIT"

    # confidence 0..100
    conf = int(100 * max(abs(p_up-0.5)*2, 0.05))
    plan = make_trade_plan(df, side)
    return {
        "time": _utcnow(),
        "side": side,
        "price": price,
        "rsi": rsi, "macd": macd, "ema": ema, "atr": atr,
        "prob_up": p_up, "orderbook_imbalance": ob_imb,
        "confidence": conf,
        "plan": plan
    }

def suggest_pairs(ex: ExchangeManager, max_pairs=12, timeframe="5m", ohlcv_limit=120):
    """
    Suggest pairs by 24h volume (quoteVolume) and realized volatility (std dev).
    """
    try:
        tickers = ex.fetch_top_tickers() or {}
    except Exception:
        tickers = {}

    # candidates = high USDT quote volume
    cands = []
    for sym, tk in tickers.items():
        if "/USDT" in sym and isinstance(tk, dict):
            qv = tk.get("quoteVolume") or tk.get("baseVolume") or 0
            cands.append((sym, float(qv)))
    cands.sort(key=lambda x: x[1], reverse=True)
    cands = [s for s,_ in cands[:max_pairs*3]] or ex.symbols_usdt(max_pairs*3)

    rows = []
    for s in cands[:max_pairs*4]:
        try:
            df = ex.fetch_ohlcv(s, timeframe=timeframe, limit=ohlcv_limit)
            if df is None or df.empty:
                continue
            vol = float(np.std(df["close"].pct_change().dropna()))
            rows.append((s, vol))
        except Exception:
            continue
    rows.sort(key=lambda x: x[1], reverse=True)
    # return top by volatility
    return [s for s,_ in rows[:max_pairs]]

def bitnodes_snapshot():
    """
    Optional BTC network context. If request fails, return None.
    """
    try:
        r = requests.get("https://bitnodes.io/api/v1/snapshots/latest/", timeout=8)
        if r.status_code == 200:
            j = r.json()
            return {
                "nodes": j.get("total_nodes"),
                "timestamp": j.get("timestamp")
            }
    except Exception:
        pass
    return None

def format_signal_for_display(sig: dict) -> str:
    lines = [
        f"**Time:** {sig['time']}",
        f"**Side:** {sig['side']}  |  **Price:** {sig['price']:.4f}",
        f"**RSI:** {sig['rsi']:.2f}  |  **MACD:** {sig['macd']:.4f}  |  **EMA:** {sig['ema']:.4f}",
        f"**Prob Up:** {sig['prob_up']:.2%}  |  **OB Imbalance:** {sig['orderbook_imbalance']:.2%}",
        f"**Confidence:** {sig['confidence']} / 100",
    ]
    if sig["plan"]["stop_loss"] and sig["plan"]["take_profit"]:
        lines.append(f"**SL:** {sig['plan']['stop_loss']:.4f}  |  **TP:** {sig['plan']['take_profit']:.4f}")
        lines.append(f"**Close by:** {sig['plan']['close_by']}")
    return "\n".join(lines)

def send_telegram(sig: dict, token: str, chat_id: str) -> bool:
    if not token or not chat_id:
        return False
    txt = (
        f"[{sig['time']}] {sig.get('pair','?')} on {sig.get('exchange','?')}\n"
        f"{sig['side']} @ {sig['price']:.4f}\n"
        f"RSI {sig['rsi']:.1f} | MACD {sig['macd']:.4f} | EMA {sig['ema']:.2f}\n"
        f"ProbUp {sig['prob_up']:.1%} | Conf {sig['confidence']}/100\n"
    )
    if sig["plan"]["stop_loss"] and sig["plan"]["take_profit"]:
        txt += f"SL {sig['plan']['stop_loss']:.4f} | TP {sig['plan']['take_profit']:.4f}\nCloseBy {sig['plan']['close_by']}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": txt})
        return r.status_code == 200
    except Exception:
        return False
