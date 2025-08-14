import os, time, math, io
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# TA
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# ML
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Exchange connector
try:
    import ccxt
    CCXT = True
except Exception:
    CCXT = False

DEFAULT_LIMIT = 500
SAFE_TIMEOUT = 20000

SUPPORTED_EXCHANGES = ["binance", "mexc", "mock"]


# -------- Utilities --------
def _utcnow():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def backoff_retry(fn, exceptions, tries=3, base_delay=0.75):
    def wrapper(*args, **kwargs):
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except exceptions as e:
                attempt += 1
                if attempt >= tries:
                    raise e
                time.sleep(base_delay * (2 ** (attempt - 1)))
    return wrapper


# -------- Exchange --------
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
        if self.mock or not self.ex:
            return [
                "BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT",
                "EUR/USD","GBP/USD","USD/JPY"
            ]
        syms = [s for s in self.ex.symbols if ("/USDT" in s) or s in ("EUR/USD","GBP/USD","USD/JPY","USD/CHF","AUD/USD","NZD/USD","USD/CAD")]
        return syms[:max_items]

    @backoff_retry
    def _fetch_ohlcv_ccxt(self, symbol, timeframe="1m", limit=DEFAULT_LIMIT):
        return self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

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
        # real ccxt with backoff
        data = self._fetch_ohlcv_ccxt(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    @backoff_retry
    def _fetch_order_book_ccxt(self, symbol, limit=50):
        return self.ex.fetch_order_book(symbol, limit=limit)

    def fetch_orderbook(self, symbol, limit=50):
        if self.mock or not self.ex:
            px = 100 + np.random.randn()*2
            bids = [[px - i*0.1, 1+np.random.rand()] for i in range(limit)]
            asks = [[px + i*0.1, 1+np.random.rand()] for i in range(limit)]
            return {"bids": bids, "asks": asks}
        return self._fetch_order_book_ccxt(symbol, limit=limit)

    def fetch_top_tickers(self):
        if self.mock or not self.ex:
            return {}
        try:
            return self.ex.fetch_tickers()
        except Exception:
            return {}


# -------- Indicators & Features --------
def analyze(df: pd.DataFrame):
    df = df.copy()
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    macd = MACD(df["close"])  # uses defaults 12,26,9
    df["macd"] = macd.macd_diff()
    df["ema"]  = EMAIndicator(df["close"], window=21).ema_indicator()
    df["atr"]  = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"].abs().replace(0, np.nan)
    # rate of change 3 bars
    df["roc3"] = (df["close"] - df["close"].shift(3)) / df["close"].shift(3).replace(0, np.nan)
    return df


def orderbook_imbalance(ob):
    if not ob or "bids" not in ob or "asks" not in ob:
        return 0.5
    bid_vol = float(sum([v for _, v in ob["bids"][:20]])) or 1e-9
    ask_vol = float(sum([v for _, v in ob["asks"][:20]])) or 1e-9
    return float(bid_vol / (bid_vol + ask_vol))


# -------- ML Probability Model (online-trained logistic regression) --------
class ProbModel:
    def __init__(self):
        # lightweight pipeline
        self.model = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(max_iter=500))
        ])
        self.is_fit = False

    def _features(self, df: pd.DataFrame, ob_imb: float) -> np.ndarray:
        x = np.array([
            df["roc3"].iloc[-1],
            float(df["close"].iloc[-1] > df["ema"].iloc[-1]),
            ob_imb,
            df["rsi"].iloc[-1] / 100.0,
            df["macd"].iloc[-1],
            df["bb_width"].iloc[-1],
        ], dtype=float).reshape(1, -1)
        return x

    def _label_series(self, df: pd.DataFrame, horizon=3) -> Tuple[np.ndarray, np.ndarray]:
        # create a simple up/down label over next "horizon" bars
        df = df.copy()
        df["fwd"] = df["close"].shift(-horizon)
        df.dropna(inplace=True)
        y = (df["fwd"] > df["close"]).astype(int).values
        # features for each row
        feats = np.stack([
            df["roc3"].values,
            (df["close"].values > df["ema"].values).astype(float),
            np.clip((df["close"].pct_change().rolling(20).sum() * 0 + 0.5).values, 0, 1),  # placeholder for OB imb (unknown historically)
            (df["rsi"].values / 100.0),
            df["macd"].values,
            df["bb_width"].values,
        ], axis=1)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats, y

    def fit_if_needed(self, df: pd.DataFrame):
        if len(df) < 60:
            return
        try:
            X, y = self._label_series(df.dropna())
            if len(np.unique(y)) < 2:
                return
            self.model.fit(X, y)
            self.is_fit = True
        except Exception:
            pass

    def predict(self, df: pd.DataFrame, ob_imb: float) -> float:
        try:
            if not self.is_fit:
                self.fit_if_needed(df)
            x = self._features(df, ob_imb)
            if self.is_fit:
                p_up = float(self.model.predict_proba(x)[0, 1])
            else:
                # fallback heuristic if not fit yet
                roc = float(df["roc3"].iloc[-1])
                trend = 1.0 if df["close"].iloc[-1] > df["ema"].iloc[-1] else 0.0
                roc01 = 1/(1+math.exp(-5*roc))
                p_up = 0.4*roc01 + 0.3*trend + 0.3*ob_imb
            return float(min(max(p_up, 0.0), 1.0))
        except Exception:
            return 0.5


# -------- Higher Timeframe Map --------
HTF_MAP = {"1m":"5m","5m":"15m","15m":"1h","1h":"4h","4h":"1d"}


def fetch_htf_alignment(ex: ExchangeManager, symbol: str, timeframe: str) -> Optional[bool]:
    htf = HTF_MAP.get(timeframe)
    if not htf:
        return None
    try:
        df_htf = ex.fetch_ohlcv(symbol, timeframe=htf, limit=120)
        df_htf = analyze(df_htf)
        price = float(df_htf["close"].iloc[-1])
        ema = float(df_htf["ema"].iloc[-1])
        return price > ema
    except Exception:
        return None


# -------- Trade Plan & Risk --------
def make_trade_plan(df: pd.DataFrame, side: str, confidence: int, equity: float, risk_pct: float):
    price = float(df["close"].iloc[-1])
    atr = float(df["atr"].iloc[-1]) if not np.isnan(df["atr"].iloc[-1]) else max(0.5, price*0.002)
    # adaptive multipliers by confidence
    mult = 1.0 + (confidence - 50)/100.0  # 0.5..1.5 roughly
    if side == "BUY":
        sl = price - 1.5*atr*mult
        tp = price + 3.0*atr*mult
    elif side == "SELL":
        sl = price + 1.5*atr*mult
        tp = price - 3.0*atr*mult
    else:
        return {"price": price, "stop_loss": None, "take_profit": None, "close_by": None, "qty": 0}

    # Position sizing: risk fixed % of equity to stop
    risk_amount = equity * (risk_pct/100.0)
    stop_distance = abs(price - sl)
    qty = float(risk_amount / max(stop_distance, 1e-9))

    close_by = (datetime.utcnow() + timedelta(minutes=45)).isoformat(timespec="seconds") + "Z"
    return {"price": price, "stop_loss": sl, "take_profit": tp, "close_by": close_by, "qty": qty}


# -------- Signal Generation --------
prob_model = ProbModel()


def generate_signal(df: pd.DataFrame, ob_imb: float, htf_align: Optional[bool]=None, atr_bounds=(0.002, 0.03),
                    equity: float=1000.0, risk_pct: float=1.0):
    df = analyze(df)
    rsi   = float(df["rsi"].iloc[-1])
    macd  = float(df["macd"].iloc[-1])
    ema   = float(df["ema"].iloc[-1])
    atr   = float(df["atr"].iloc[-1])
    price = float(df["close"].iloc[-1])

    # probability
    p_up  = prob_model.predict(df, ob_imb)

    # filters
    atr_pct = atr / max(price, 1e-9)
    atr_ok = (atr_bounds[0] <= atr_pct <= atr_bounds[1])

    # signal smoothing: require last 2 bars agree
    df_tail = df.tail(3)
    bullish_bars = int((df_tail["close"] > df_tail["ema"]).sum())
    bearish_bars = int((df_tail["close"] < df_tail["ema"]).sum())

    long_ok  = (rsi < 55 and macd > 0 and price > ema and p_up > 0.55 and atr_ok and bullish_bars >= 2)
    short_ok = (rsi > 45 and macd < 0 and price < ema and p_up < 0.45 and atr_ok and bearish_bars >= 2)

    # higher timeframe confirmation if available
    if htf_align is True:
        short_ok = False if long_ok else short_ok  # favor longs
    elif htf_align is False:
        long_ok = False if short_ok else long_ok  # favor shorts

    if long_ok and not short_ok:
        side = "BUY"
    elif short_ok and not long_ok:
        side = "SELL"
    else:
        side = "WAIT"

    # confidence 0..100 from p_up distance + bb width (vol expansion bonus)
    conf_core = max(abs(p_up-0.5)*2, 0.05)
    bb_bonus = float(np.clip(df["bb_width"].iloc[-1] / (df["bb_width"].rolling(100).median().iloc[-1] + 1e-9), 0.8, 1.2) - 1.0)
    confidence = int(100 * np.clip(conf_core + 0.2*bb_bonus, 0.05, 1.0))

    plan = make_trade_plan(df, side, confidence, equity, risk_pct)
    return {
        "time": _utcnow(),
        "side": side,
        "price": price,
        "rsi": rsi, "macd": macd, "ema": ema, "atr": atr,
        "prob_up": p_up, "orderbook_imbalance": ob_imb,
        "confidence": confidence,
        "plan": plan
    }


# -------- Pair Suggestions (Volume + Volatility + Breakout) --------
def suggest_pairs(ex: ExchangeManager, max_pairs=12, timeframe="5m", ohlcv_limit=200):
    try:
        tickers = ex.fetch_top_tickers() or {}
    except Exception:
        tickers = {}

    cands = []
    for sym, tk in tickers.items():
        if "/USDT" in sym and isinstance(tk, dict):
            qv = tk.get("quoteVolume") or tk.get("baseVolume") or 0
            cands.append((sym, float(qv)))
    cands.sort(key=lambda x: x[1], reverse=True)
    cands = [s for s,_ in cands[:max_pairs*4]] or ex.symbols_usdt(max_pairs*4)

    rows = []
    for s in cands[:max_pairs*6]:
        try:
            df = ex.fetch_ohlcv(s, timeframe=timeframe, limit=ohlcv_limit)
            if df is None or df.empty:
                continue
            df = analyze(df)
            vol = float(np.std(df["close"].pct_change().dropna()))
            # breakout score: recent BB width expansion vs 100-bar median
            bbw = float(df["bb_width"].iloc[-1])
            med = float(df["bb_width"].rolling(100).median().iloc[-1]) if len(df) >= 100 else float(np.median(df["bb_width"].values))
            breakout = (bbw / (med + 1e-9))
            score = 0.6*vol + 0.4*breakout
            rows.append((s, score))
        except Exception:
            continue
    rows.sort(key=lambda x: x[1], reverse=True)
    return [s for s,_ in rows[:max_pairs]]


# -------- External Context --------
def bitnodes_snapshot():
    try:
        r = requests.get("https://bitnodes.io/api/v1/snapshots/latest/", timeout=8)
        if r.status_code == 200:
            j = r.json()
            return {"nodes": j.get("total_nodes"), "timestamp": j.get("timestamp")}
    except Exception:
        pass
    return None


# -------- Formatting & Telegram --------
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
        lines.append(f"**Qty:** {sig['plan']['qty']:.4f}")
        lines.append(f"**Close by:** {sig['plan']['close_by']}")
    return "\n".join(lines)


def _plot_candles_image(df: pd.DataFrame) -> Optional[bytes]:
    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Candlestick(x=df["datetime"], open=df["open"], high=df["high"], low=df["low"], close=df["close"] )])
        fig.update_layout(height=380, margin=dict(l=10,r=10,t=20,b=10))
        buf = io.BytesIO()
        fig.write_image(buf, format="png")  # requires kaleido
        return buf.getvalue()
    except Exception:
        return None


def send_telegram(sig: dict, token: str, chat_id: str, df: Optional[pd.DataFrame]=None) -> bool:
    if not token or not chat_id:
        return False
    txt = (
        f"[{sig['time']}] {sig.get('pair','?')} on {sig.get('exchange','?')}\n"
        f"{sig['side']} @ {sig['price']:.4f}\n"
        f"RSI {sig['rsi']:.1f} | MACD {sig['macd']:.4f} | EMA {sig['ema']:.2f}\n"
        f"ProbUp {sig['prob_up']:.1%} | Conf {sig['confidence']}/100\n"
    )
    if sig["plan"]["stop_loss"] and sig["plan"]["take_profit"]:
        txt += f"SL {sig['plan']['stop_loss']:.4f} | TP {sig['plan']['take_profit']:.4f} | Qty {sig['plan']['qty']:.4f}\nCloseBy {sig['plan']['close_by']}\n"

    url_send_text = f"https://api.telegram.org/bot{token}/sendMessage"
    url_send_photo = f"https://api.telegram.org/bot{token}/sendPhoto"
    ok = True
    try:
        r = requests.post(url_send_text, data={"chat_id": chat_id, "text": txt})
        ok = ok and (r.status_code == 200)
    except Exception:
        ok = False

    # try send chart image
    if df is not None:
        img = _plot_candles_image(df)
        if img is not None:
            files = {"photo": ("chart.png", img, "image/png")}
            try:
                r2 = requests.post(url_send_photo, data={"chat_id": chat_id, "caption": "Signal chart"}, files=files)
                ok = ok and (r2.status_code == 200)
            except Exception:
                pass
    return ok


# -------- Backtesting --------
def run_backtest(ex: ExchangeManager, symbol: str, timeframe="5m", lookback=1000, equity=1000.0, risk_pct=1.0):
    df = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback)
    df = analyze(df)
    balance = equity
    equity_curve = []

    position = None  # {side, entry, sl, tp, qty}

    for i in range(60, len(df)-1):
        window = df.iloc[:i].copy()
        ob_imb = 0.5  # historical OB unknown
        htf_align = None
        sig = generate_signal(window, ob_imb, htf_align, equity=balance, risk_pct=risk_pct)
        sig["pair"] = symbol
        sig["exchange"] = ex.name

        # close position if SL/TP hit on the next bar (simplification)
        if position is not None:
            nxt = df.iloc[i+1]
            if position["side"] == "BUY":
                if nxt.low <= position["sl"]:
                    balance -= position["qty"] * (position["entry"] - position["sl"])  # loss
                    position = None
                elif nxt.high >= position["tp"]:
                    balance += position["qty"] * (position["tp"] - position["entry"])  # win
                    position = None
            else:  # SELL
                if nxt.high >= position["sl"]:
                    balance -= position["qty"] * (position["sl"] - position["entry"])  # loss
                    position = None
                elif nxt.low <= position["tp"]:
                    balance += position["qty"] * (position["entry"] - position["tp"])  # win
                    position = None

        # open new position if WAIT -> skip
        if sig["side"] != "WAIT" and position is None:
            plan = sig["plan"]
            position = {
                "side": sig["side"],
                "entry": sig["price"],
                "sl": plan["stop_loss"],
                "tp": plan["take_profit"],
                "qty": plan["qty"],
            }

        equity_curve.append(balance)

    # metrics
    curve = np.array(equity_curve)
    if len(curve) == 0:
        return {"final_equity": balance, "returns": 0.0, "max_dd": 0.0, "trades": 0}
    returns = (curve[-1] - equity) / equity
    peak = np.maximum.accumulate(curve)
    drawdown = (curve - peak) / peak
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    return {"final_equity": float(curve[-1]), "returns": float(returns), "max_dd": float(max_dd), "trades": int(np.sum(np.diff(curve) != 0))}

